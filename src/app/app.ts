import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  OnInit,
  ViewChild,
  computed,
  inject,
  signal
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { EyeTrackerService } from './services/eye-tracker.service';
import { ChallengeVideosService, ChallengePreset } from './services/challenge-videos.service';
import { CompositeRecorderService, ReticleStyle, PipPosition } from './services/composite-recorder.service';
import { VoiceCalibratorService } from './services/voice-calibrator.service';

type ActiveTab = 'studio' | 'calibration' | 'review' | 'diagnostics';

@Component({
  selector: 'app-root',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, MatIconModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, OnDestroy {
  readonly eyeTracker = inject(EyeTrackerService);
  readonly challengeVideos = inject(ChallengeVideosService);
  readonly recorder = inject(CompositeRecorderService);
  readonly voiceCalibrator = inject(VoiceCalibratorService);

  @ViewChild('studioCanvas', { static: false }) studioCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('fileInput', { static: false }) fileInputRef!: ElementRef<HTMLInputElement>;

  // Tabs
  readonly activeTab = signal<ActiveTab>('studio');

  // Studio render animation frame
  private renderLoopId: number | null = null;

  // Computed properties from services
  readonly isRecording = computed(() => this.recorder.isRecording());
  readonly isCameraReady = computed(() => this.eyeTracker.isCameraReady());
  readonly isSimulatedGaze = computed(() => this.eyeTracker.isSimulatedGaze());
  readonly currentGaze = computed(() => this.eyeTracker.currentGaze());
  readonly activePreset = computed(() => this.challengeVideos.activePreset());
  readonly presets = computed(() => this.challengeVideos.presets);
  readonly isPlaying = computed(() => this.challengeVideos.isPlaying());
  readonly currentTime = computed(() => this.challengeVideos.currentTime());
  readonly duration = computed(() => this.challengeVideos.duration());
  readonly recordedVideoUrl = computed(() => this.recorder.recordedVideoUrl());
  readonly heatmapImageUrl = computed(() => this.recorder.heatmapImageUrl());
  readonly telemetryLog = computed(() => this.recorder.getTelemetryLog());

  // Calibration and voice state
  readonly voiceTargets = computed(() => this.voiceCalibrator.targets);
  readonly currentVoiceTargetIndex = computed(() => this.voiceCalibrator.currentTargetIndex());
  readonly voiceStatusMessage = computed(() => this.voiceCalibrator.statusMessage());
  readonly lastHeardPhrase = computed(() => this.voiceCalibrator.lastHeardPhrase());
  readonly micVolume = computed(() => this.voiceCalibrator.micVolume());
  readonly isVoiceSpeaking = computed(() => this.voiceCalibrator.isSpeaking());
  readonly isListeningForVoice = computed(() => this.voiceCalibrator.isListeningForVoice());
  readonly isVoiceSupported = computed(() => this.voiceCalibrator.isVoiceSupported());
  readonly isCalibrated = computed(() => this.eyeTracker.isCalibrated());
  readonly calibrationAccuracy = computed(() => this.eyeTracker.calibrationAccuracy());
  readonly pupilConfidence = computed(() => this.eyeTracker.pupilConfidence());
  readonly dwellProgress = computed(() => this.voiceCalibrator.dwellProgress());
  readonly modelStatus = computed(() => this.eyeTracker.modelStatus());
  readonly isNeuralModelLoaded = computed(() => this.eyeTracker.isNeuralModelLoaded());

  readonly formattedTime = computed(() => {
    const sec = this.currentTime();
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  });

  readonly formattedDuration = computed(() => {
    const sec = this.duration();
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  });

  readonly formattedRecTime = computed(() => {
    const sec = this.recorder.recordingTimeSeconds();
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
  });

  async ngOnInit() {
    try {
      if (typeof navigator !== 'undefined' && navigator.mediaDevices) {
        await this.eyeTracker.startWebcam();
      } else {
        this.eyeTracker.setSimulatedGaze(true);
      }
    } catch {
      this.eyeTracker.setSimulatedGaze(true);
    }

    this.startStudioCanvasLoop();
  }

  ngOnDestroy() {
    if (this.renderLoopId) {
      cancelAnimationFrame(this.renderLoopId);
    }
    this.voiceCalibrator.cancelCalibration();
    this.eyeTracker.stopWebcam();
  }

  @HostListener('window:keydown', ['$event'])
  onKeyDown(event: KeyboardEvent) {
    if (this.activeTab() === 'calibration') {
      if (event.code === 'Space' || event.code === 'Enter') {
        event.preventDefault();
        this.confirmCurrentCalibrationTarget();
      }
    }
  }

  setTab(tab: ActiveTab) {
    this.activeTab.set(tab);

    if (tab === 'calibration') {
      this.startVoiceCalibration();
    } else {
      this.voiceCalibrator.cancelCalibration();
    }
  }

  startVoiceCalibration() {
    this.eyeTracker.startCalibration();
    this.voiceCalibrator.startCalibrationSession(
      (targetIndex) => {
        this.eyeTracker.recordCalibrationPoint(targetIndex);
      },
      () => {
        // Automatically return to studio after voice completion
        setTimeout(() => {
          this.activeTab.set('studio');
        }, 1500);
      }
    );
  }

  confirmCurrentCalibrationTarget() {
    this.voiceCalibrator.confirmCurrentTargetManual();
  }

  handleCalibrationPointClick(targetIndex: number) {
    if (targetIndex === this.currentVoiceTargetIndex()) {
      this.voiceCalibrator.manualConfirmTarget(targetIndex);
    }
  }

  async toggleWebcam() {
    if (this.eyeTracker.isCameraReady()) {
      this.eyeTracker.stopWebcam();
      this.eyeTracker.setSimulatedGaze(true);
    } else {
      try {
        await this.eyeTracker.startWebcam();
      } catch (e) {
        console.error('Failed to enable camera', e);
      }
    }
  }

  toggleSimulatedGaze() {
    const newState = !this.eyeTracker.isSimulatedGaze();
    this.eyeTracker.setSimulatedGaze(newState);
  }

  onStageMouseMove(event: MouseEvent) {
    if (!this.eyeTracker.isSimulatedGaze()) return;

    const target = event.currentTarget as HTMLElement;
    if (!target) return;

    const rect = target.getBoundingClientRect();
    const normX = (event.clientX - rect.left) / rect.width;
    const normY = (event.clientY - rect.top) / rect.height;

    this.eyeTracker.updateSimulatedGaze(normX, normY, 1280, 720);
  }

  onStageTouchMove(event: TouchEvent) {
    if (!this.eyeTracker.isSimulatedGaze()) return;
    const touch = event.touches[0];
    if (!touch) return;

    const target = event.currentTarget as HTMLElement;
    if (!target) return;

    const rect = target.getBoundingClientRect();
    const normX = (touch.clientX - rect.left) / rect.width;
    const normY = (touch.clientY - rect.top) / rect.height;

    this.eyeTracker.updateSimulatedGaze(normX, normY, 1280, 720);
  }

  togglePlayPause() {
    if (this.isPlaying()) {
      this.challengeVideos.pause();
    } else {
      this.challengeVideos.play();
    }
  }

  resetChallenge() {
    this.challengeVideos.stop();
  }

  selectPreset(preset: ChallengePreset) {
    this.challengeVideos.selectPreset(preset);
  }

  triggerUpload() {
    if (this.fileInputRef) {
      this.fileInputRef.nativeElement.click();
    }
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.challengeVideos.loadCustomVideoFile(input.files[0]);
    }
  }

  setReticleStyle(style: ReticleStyle) {
    this.recorder.reticleStyle.set(style);
  }

  setPipPosition(pos: PipPosition) {
    this.recorder.pipPosition.set(pos);
  }

  setSmoothing(event: Event) {
    const target = event.target as HTMLInputElement;
    const val = parseFloat(target.value);
    this.eyeTracker.setSmoothing(val);
  }

  toggleGazeTrail() {
    this.recorder.showGazeTrail.set(!this.recorder.showGazeTrail());
  }

  toggleTelemetryHud() {
    this.recorder.showTelemetryHud.set(!this.recorder.showTelemetryHud());
  }

  toggleEyeCropZoom() {
    this.recorder.showEyeCropZoom.set(!this.recorder.showEyeCropZoom());
  }

  async toggleRecording() {
    if (this.isRecording()) {
      this.recorder.stopRecording();
      this.activeTab.set('review');
    } else {
      await this.recorder.startRecording();
    }
  }

  downloadReactionVideo() {
    const url = this.recordedVideoUrl();
    if (!url) return;

    const a = document.createElement('a');
    a.href = url;
    a.download = `gazecast_reaction_${Date.now()}.webm`;
    a.click();
  }

  downloadHeatmap() {
    const url = this.heatmapImageUrl();
    if (!url) return;

    const a = document.createElement('a');
    a.href = url;
    a.download = `gazecast_attention_heatmap_${Date.now()}.png`;
    a.click();
  }

  exportCsv() {
    this.recorder.exportTelemetryCsv();
  }

  exportJson() {
    this.recorder.exportTelemetryJson();
  }

  private startStudioCanvasLoop() {
    if (typeof window === 'undefined') return;

    const draw = () => {
      this.recorder.renderCompositeFrame();

      if (this.studioCanvasRef && this.studioCanvasRef.nativeElement) {
        const displayCanvas = this.studioCanvasRef.nativeElement;
        const dCtx = displayCanvas.getContext('2d');
        const compCanvas = this.recorder.getCompositorCanvas();
        if (dCtx && compCanvas) {
          dCtx.drawImage(compCanvas, 0, 0, displayCanvas.width, displayCanvas.height);
        }
      }

      this.renderLoopId = requestAnimationFrame(draw);
    };

    this.renderLoopId = requestAnimationFrame(draw);
  }
}
