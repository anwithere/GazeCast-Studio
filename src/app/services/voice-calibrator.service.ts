import { Injectable, signal } from '@angular/core';

export interface CalibrationVoiceTarget {
  id: number;
  label: string;
  spokenPrompt: string;
  nx: number;
  ny: number;
  completed: boolean;
}

interface SpeechRecognitionResultItem {
  transcript: string;
}

type SpeechRecognitionResultList = Record<number, Record<number, SpeechRecognitionResultItem>> & {
  length: number;
};

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEventLike {
  error: string;
}

interface SpeechRecognitionInstance {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

@Injectable({
  providedIn: 'root'
})
export class VoiceCalibratorService {
  // Speech & Recognition state
  readonly isVoiceSupported = signal<boolean>(true);
  readonly isSpeechRecognitionActive = signal<boolean>(false);
  readonly isSpeaking = signal<boolean>(false);
  readonly lastHeardPhrase = signal<string>('');
  readonly micVolume = signal<number>(0); // 0 to 100
  readonly statusMessage = signal<string>('Ready for voice calibration');
  readonly isListeningForVoice = signal<boolean>(false);
  readonly dwellProgress = signal<number>(0); // 0 to 100%

  // 9 Targets
  readonly targets: CalibrationVoiceTarget[] = [
    { id: 0, label: 'Top Left', spokenPrompt: 'Look at top left, and say: I have looked', nx: 0.12, ny: 0.12, completed: false },
    { id: 1, label: 'Top Center', spokenPrompt: 'Look at top center, and say: I have looked', nx: 0.50, ny: 0.12, completed: false },
    { id: 2, label: 'Top Right', spokenPrompt: 'Look at top right, and say: I have looked', nx: 0.88, ny: 0.12, completed: false },
    { id: 3, label: 'Middle Left', spokenPrompt: 'Look at middle left, and say: I have looked', nx: 0.12, ny: 0.50, completed: false },
    { id: 4, label: 'Center', spokenPrompt: 'Look directly at center, and say: I have looked', nx: 0.50, ny: 0.50, completed: false },
    { id: 5, label: 'Middle Right', spokenPrompt: 'Look at middle right, and say: I have looked', nx: 0.88, ny: 0.50, completed: false },
    { id: 6, label: 'Bottom Left', spokenPrompt: 'Look at bottom left, and say: I have looked', nx: 0.12, ny: 0.88, completed: false },
    { id: 7, label: 'Bottom Center', spokenPrompt: 'Look at bottom center, and say: I have looked', nx: 0.50, ny: 0.88, completed: false },
    { id: 8, label: 'Bottom Right', spokenPrompt: 'Look at bottom right, and say: I have looked', nx: 0.88, ny: 0.88, completed: false },
  ];

  readonly currentTargetIndex = signal<number>(0);
  readonly isCalibrating = signal<boolean>(false);
  readonly calibrationFinished = signal<boolean>(false);

  // Callbacks
  private onTargetConfirmedCallback: ((targetIndex: number) => void) | null = null;
  private onAllCompletedCallback: (() => void) | null = null;

  // Audio & Speech instances
  private recognition: SpeechRecognitionInstance | null = null;
  private synth: SpeechSynthesis | null = null;
  private audioCtx: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private micAnalyser: AnalyserNode | null = null;
  private animFrameId: number | null = null;

  // Vocal Energy (VAD) detection parameters
  private isProcessingConfirmation = false;
  private voiceEnergyStartTime = 0;
  private isVoiceActive = false;
  private dwellInterval: ReturnType<typeof setInterval> | null = null;
  private dwellCounter = 0;

  constructor() {
    if (typeof window !== 'undefined') {
      if ('speechSynthesis' in window) {
        this.synth = window.speechSynthesis;
      }

      const win = window as unknown as {
        SpeechRecognition?: SpeechRecognitionConstructor;
        webkitSpeechRecognition?: SpeechRecognitionConstructor;
      };

      const SpeechRec = win.SpeechRecognition || win.webkitSpeechRecognition;
      if (SpeechRec) {
        this.initRecognition(SpeechRec);
      }
    }
  }

  private initRecognition(SpeechRec: SpeechRecognitionConstructor) {
    try {
      this.recognition = new SpeechRec();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = 'en-US';

      this.recognition.onstart = () => {
        this.isSpeechRecognitionActive.set(true);
        this.isListeningForVoice.set(true);
      };

      this.recognition.onresult = (event: SpeechRecognitionEventLike) => {
        if (this.isSpeaking()) return; // Don't trigger on own voice
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const item = event.results[i]?.[0];
          if (item) {
            transcript += item.transcript.toLowerCase();
          }
        }

        this.lastHeardPhrase.set(transcript.trim());

        // Lenient matching: matches "looked", "look", "i have", "done", "ready", "next", "ok", "yes"
        if (
          transcript.includes('looked') ||
          transcript.includes('look') ||
          transcript.includes('have') ||
          transcript.includes('done') ||
          transcript.includes('ready') ||
          transcript.includes('next') ||
          transcript.includes('yes') ||
          transcript.includes('ok')
        ) {
          this.handleHeardConfirmation('Speech Recognition');
        }
      };

      this.recognition.onerror = (e: SpeechRecognitionErrorEventLike) => {
        console.warn('Speech recognition warning:', e.error);
      };

      this.recognition.onend = () => {
        this.isSpeechRecognitionActive.set(false);
        if (this.isCalibrating() && !this.calibrationFinished()) {
          try {
            this.recognition?.start();
          } catch {
            // Ignore
          }
        }
      };
    } catch (e) {
      console.warn('Error setting up speech recognition:', e);
    }
  }

  startCalibrationSession(
    onConfirm: (targetIndex: number) => void,
    onComplete: () => void
  ) {
    this.onTargetConfirmedCallback = onConfirm;
    this.onAllCompletedCallback = onComplete;
    this.isCalibrating.set(true);
    this.calibrationFinished.set(false);
    this.currentTargetIndex.set(0);
    this.targets.forEach(t => (t.completed = false));
    this.isProcessingConfirmation = false;
    this.dwellProgress.set(0);

    // 1. Start real-time microphone energy monitor (Web Audio VAD)
    this.startMicVisualizer();

    // 2. Start speech recognition in background
    this.startListening();

    // 3. Start auto-dwell timer (auto-captures after 3 seconds of gaze fixation if user prefers not speaking)
    this.startDwellTimer();

    // 4. Welcome vocal prompt
    const firstTarget = this.targets[0];
    this.speak(
      `Please look directly at the ${firstTarget.label} target, and say: I have looked.`,
      () => {
        this.statusMessage.set(`Looking at ${firstTarget.label}... Say "I have looked" or tap Spacebar`);
      }
    );
  }

  private startDwellTimer() {
    if (this.dwellInterval) clearInterval(this.dwellInterval);
    this.dwellCounter = 0;
    this.dwellProgress.set(0);

    this.dwellInterval = setInterval(() => {
      if (!this.isCalibrating() || this.isProcessingConfirmation || this.isSpeaking()) return;

      this.dwellCounter += 100;
      const progress = Math.min(100, Math.round((this.dwellCounter / 3200) * 100));
      this.dwellProgress.set(progress);

      // Auto-advance after 3.2s of steady fixation
      if (this.dwellCounter >= 3200) {
        this.handleHeardConfirmation('Fixation Timer');
      }
    }, 100);
  }

  private resetDwell() {
    this.dwellCounter = 0;
    this.dwellProgress.set(0);
  }

  handleHeardConfirmation(source = 'Vocal') {
    if (this.isProcessingConfirmation || !this.isCalibrating() || this.calibrationFinished()) {
      return;
    }

    this.isProcessingConfirmation = true;
    this.resetDwell();

    const currentIndex = this.currentTargetIndex();
    const currentTarget = this.targets[currentIndex];

    if (!currentTarget) return;

    this.statusMessage.set(`Target ${currentIndex + 1} locked via ${source}!`);

    // High confirmation chime
    this.playChime(true);
    currentTarget.completed = true;

    if (this.onTargetConfirmedCallback) {
      this.onTargetConfirmedCallback(currentIndex);
    }

    const nextIndex = currentIndex + 1;
    if (nextIndex < this.targets.length) {
      this.currentTargetIndex.set(nextIndex);
      const nextTarget = this.targets[nextIndex];

      const confirmations = [
        `Target captured! Now look at ${nextTarget.label}, and say: I have looked.`,
        `Got it! Look at ${nextTarget.label}.`,
        `Recorded! Focus on ${nextTarget.label}.`
      ];
      const prompt = confirmations[currentIndex % confirmations.length];

      this.speak(prompt, () => {
        this.statusMessage.set(`Looking at ${nextTarget.label}... Say "I have looked" or tap Spacebar`);
        this.isProcessingConfirmation = false;
        this.resetDwell();
      });
    } else {
      // Completed all 9 targets!
      this.calibrationFinished.set(true);
      this.isCalibrating.set(false);
      this.statusMessage.set('Calibration complete! High accuracy gaze model locked.');

      if (this.dwellInterval) clearInterval(this.dwellInterval);

      this.playFanfare();
      this.speak('Calibration complete! Your screen gaze mapping is now active.', () => {
        this.stopListening();
        this.stopMicVisualizer();
        if (this.onAllCompletedCallback) {
          this.onAllCompletedCallback();
        }
      });
    }
  }

  manualConfirmTarget(targetIndex: number) {
    if (targetIndex !== this.currentTargetIndex()) return;
    this.handleHeardConfirmation('Manual Tap');
  }

  confirmCurrentTargetManual() {
    this.handleHeardConfirmation('Spacebar Tap');
  }

  private startListening() {
    if (!this.recognition) return;
    try {
      this.recognition.start();
    } catch {
      // Ignore
    }
  }

  private stopListening() {
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch {
        // Ignore
      }
    }
    this.isSpeechRecognitionActive.set(false);
    this.isListeningForVoice.set(false);
  }

  speak(text: string, onEnd?: () => void) {
    if (!this.synth) {
      if (onEnd) onEnd();
      return;
    }

    try {
      this.synth.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      const voices = this.synth.getVoices();
      const preferredVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('David')));
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      this.isSpeaking.set(true);

      utterance.onend = () => {
        this.isSpeaking.set(false);
        if (onEnd) onEnd();
      };

      utterance.onerror = () => {
        this.isSpeaking.set(false);
        if (onEnd) onEnd();
      };

      this.synth.speak(utterance);
    } catch {
      this.isSpeaking.set(false);
      if (onEnd) onEnd();
    }
  }

  /**
   * Real-Time Web Audio VAD (Voice Activity Detection)
   * Analyzes sound energy in browser: when user speaks into mic, triggers capture instantly!
   */
  private async startMicVisualizer() {
    try {
      if (typeof window === 'undefined') return;
      if (!this.audioCtx) {
        const win = window as unknown as {
          AudioContext?: typeof AudioContext;
          webkitAudioContext?: typeof AudioContext;
        };
        const AudioContextClass = win.AudioContext || win.webkitAudioContext;
        if (AudioContextClass) {
          this.audioCtx = new AudioContextClass();
        }
      }

      if (this.audioCtx && this.audioCtx.state === 'suspended') {
        await this.audioCtx.resume();
      }

      if (!this.micStream && navigator.mediaDevices) {
        this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      }

      if (this.micStream && this.audioCtx) {
        const source = this.audioCtx.createMediaStreamSource(this.micStream);
        this.micAnalyser = this.audioCtx.createAnalyser();
        this.micAnalyser.fftSize = 64;
        source.connect(this.micAnalyser);

        const dataArray = new Uint8Array(this.micAnalyser.frequencyBinCount);
        const updateLevel = () => {
          if (!this.micAnalyser) return;
          this.micAnalyser.getByteFrequencyData(dataArray);

          let sum = 0;
          for (const val of dataArray) {
            sum += val;
          }
          const avg = sum / dataArray.length;
          const vol = Math.min(100, Math.round((avg / 128) * 100));
          this.micVolume.set(vol);

          // Voice Activity Detection (VAD)
          // When not speaking itself, if user vocalizes (vol > 20 for >= 250ms then finishes)
          const now = performance.now();
          if (!this.isSpeaking() && !this.isProcessingConfirmation && this.isCalibrating()) {
            if (vol > 22) {
              if (!this.isVoiceActive) {
                this.isVoiceActive = true;
                this.voiceEnergyStartTime = now;
              }
            } else if (this.isVoiceActive) {
              const spokenDuration = now - this.voiceEnergyStartTime;
              this.isVoiceActive = false;
              // If user spoke for between 250ms and 2200ms (typical "I have looked" or "done")
              if (spokenDuration > 250 && spokenDuration < 2500) {
                this.lastHeardPhrase.set('Vocal utterance detected');
                this.handleHeardConfirmation('Acoustic Voice (VAD)');
              }
            }
          }

          if (this.isCalibrating()) {
            this.animFrameId = requestAnimationFrame(updateLevel);
          }
        };

        this.animFrameId = requestAnimationFrame(updateLevel);
      }
    } catch (err) {
      console.warn('Mic visualizer not accessible:', err);
    }
  }

  private stopMicVisualizer() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach(t => t.stop());
      this.micStream = null;
    }
    if (this.dwellInterval) {
      clearInterval(this.dwellInterval);
      this.dwellInterval = null;
    }
    this.micVolume.set(0);
    this.dwellProgress.set(0);
  }

  private playChime(success = true) {
    try {
      if (!this.audioCtx) {
        const win = window as unknown as {
          AudioContext?: typeof AudioContext;
          webkitAudioContext?: typeof AudioContext;
        };
        const AudioContextClass = win.AudioContext || win.webkitAudioContext;
        if (AudioContextClass) {
          this.audioCtx = new AudioContextClass();
        }
      }
      if (!this.audioCtx) return;
      const ctx = this.audioCtx;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.connect(gain);
      gain.connect(ctx.destination);

      const now = ctx.currentTime;
      if (success) {
        osc.frequency.setValueAtTime(1318.5, now);
        osc.frequency.setValueAtTime(1760.0, now + 0.08);
        gain.gain.setValueAtTime(0.18, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
        osc.start(now);
        osc.stop(now + 0.35);
      } else {
        osc.frequency.setValueAtTime(440, now);
        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.start(now);
        osc.stop(now + 0.2);
      }
    } catch {
      // Audio context might be restricted
    }
  }

  private playFanfare() {
    try {
      if (!this.audioCtx) {
        const win = window as unknown as {
          AudioContext?: typeof AudioContext;
          webkitAudioContext?: typeof AudioContext;
        };
        const AudioContextClass = win.AudioContext || win.webkitAudioContext;
        if (AudioContextClass) {
          this.audioCtx = new AudioContextClass();
        }
      }
      if (!this.audioCtx) return;
      const ctx = this.audioCtx;
      const now = ctx.currentTime;
      [523.25, 659.25, 783.99, 1046.5].forEach((freq, idx) => {
        if (!ctx) return;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        const startTime = now + idx * 0.1;
        osc.frequency.setValueAtTime(freq, startTime);
        gain.gain.setValueAtTime(0.14, startTime);
        gain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.3);
        osc.start(startTime);
        osc.stop(startTime + 0.3);
      });
    } catch {
      // Ignore
    }
  }

  cancelCalibration() {
    this.isCalibrating.set(false);
    this.calibrationFinished.set(false);
    this.stopListening();
    this.stopMicVisualizer();
    if (this.synth) {
      this.synth.cancel();
    }
  }
}
