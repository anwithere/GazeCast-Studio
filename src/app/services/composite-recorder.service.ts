import { Injectable, inject, signal } from '@angular/core';
import { EyeTrackerService, GazePoint } from './eye-tracker.service';
import { ChallengeVideosService } from './challenge-videos.service';

export interface TelemetryPoint {
  timeMs: number;
  normX: number;
  normY: number;
  screenX: number;
  screenY: number;
  fixationDurationMs: number;
  confidence: number;
}

export type PipPosition = 'top-right' | 'top-left' | 'top-center' | 'split';
export type ReticleStyle = 'neon-reticle' | 'laser-pointer' | 'heat-focus' | 'bullseye';

@Injectable({
  providedIn: 'root'
})
export class CompositeRecorderService {
  private readonly eyeTracker = inject(EyeTrackerService);
  private readonly challengeVideos = inject(ChallengeVideosService);

  readonly isRecording = signal<boolean>(false);
  readonly recordedVideoUrl = signal<string | null>(null);
  readonly recordedVideoBlob = signal<Blob | null>(null);
  readonly recordedDurationMs = signal<number>(0);
  readonly recordingTimeSeconds = signal<number>(0);
  readonly heatmapImageUrl = signal<string | null>(null);
  readonly aiInsights = signal<{
    verdictTitle: string;
    focusScore: number;
    reactionSpeedMs: number;
    distractionResistance: string;
    highlights: string[];
    viralRoastOrPraise: string;
  } | null>(null);

  // Customization options
  readonly pipPosition = signal<PipPosition>('top-right');
  readonly reticleStyle = signal<ReticleStyle>('neon-reticle');
  readonly reticleColor = signal<string>('#f43f5e'); // Rose
  readonly showGazeTrail = signal<boolean>(true);
  readonly showTelemetryHud = signal<boolean>(true);
  readonly showEyeCropZoom = signal<boolean>(true);

  // Recorded telemetry points for heatmap and CSV export
  private telemetryLog: TelemetryPoint[] = [];

  // Compositor Canvas (1280x720 60FPS)
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private animId: number | null = null;

  // MediaRecorder references
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private recordingStartTime = 0;
  private timerInterval: ReturnType<typeof setInterval> | null = null;

  // Trail buffer
  private gazeTrailHistory: { x: number; y: number; time: number }[] = [];

  constructor() {
    if (typeof document !== 'undefined') {
      this.canvas = document.createElement('canvas');
      this.canvas.width = 1280;
      this.canvas.height = 720;
      this.ctx = this.canvas.getContext('2d', { alpha: false });
    }
  }

  getCompositorCanvas(): HTMLCanvasElement | null {
    return this.canvas;
  }

  getTelemetryLog(): TelemetryPoint[] {
    return this.telemetryLog;
  }

  async startRecording(): Promise<void> {
    this.telemetryLog = [];
    this.recordedChunks = [];
    this.gazeTrailHistory = [];
    this.recordedVideoUrl.set(null);
    this.recordedVideoBlob.set(null);
    this.heatmapImageUrl.set(null);

    // Start challenge video playback
    this.challengeVideos.play();

    if (!this.canvas) return;

    // Setup MediaStream from canvas
    const canvasStream = this.canvas.captureStream(60);

    // Mix microphone audio if available
    const webcamStream = this.eyeTracker.getMediaStream();
    if (webcamStream) {
      const audioTracks = webcamStream.getAudioTracks();
      if (audioTracks.length > 0) {
        canvasStream.addTrack(audioTracks[0]);
      }
    }

    // Determine supported mimeType
    let mimeType = 'video/webm;codecs=vp9,opus';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = 'video/webm';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = '';
      }
    }

    try {
      this.mediaRecorder = mimeType ? new MediaRecorder(canvasStream, { mimeType }) : new MediaRecorder(canvasStream);
    } catch {
      this.mediaRecorder = new MediaRecorder(canvasStream);
    }

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        this.recordedChunks.push(e.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      const blob = new Blob(this.recordedChunks, { type: this.mediaRecorder?.mimeType || 'video/webm' });
      this.recordedVideoBlob.set(blob);
      const url = URL.createObjectURL(blob);
      this.recordedVideoUrl.set(url);
      this.generateAttentionHeatmap();
    };

    this.mediaRecorder.start(200); // 200ms slices
    this.recordingStartTime = performance.now();
    this.isRecording.set(true);

    this.timerInterval = setInterval(() => {
      const elapsedSec = (performance.now() - this.recordingStartTime) / 1000;
      this.recordingTimeSeconds.set(Math.round(elapsedSec));
      this.recordedDurationMs.set(Math.round(performance.now() - this.recordingStartTime));
    }, 100);

    this.startCompositorLoop();
  }

  stopRecording(): void {
    if (!this.isRecording()) return;

    this.isRecording.set(false);
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    this.challengeVideos.pause();
  }

  private startCompositorLoop(): void {
    const render = () => {
      this.renderCompositeFrame();
      if (this.isRecording()) {
        this.animId = requestAnimationFrame(render);
      }
    };

    if (this.animId) cancelAnimationFrame(this.animId);
    this.animId = requestAnimationFrame(render);
  }

  /**
   * Renders the real-time composite frame
   */
  renderCompositeFrame(): void {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const now = performance.now();
    const elapsedSessionMs = this.isRecording() ? now - this.recordingStartTime : 0;

    // 1. Draw Background Stimulus Video
    this.challengeVideos.drawCurrentFrame(ctx, w, h);

    // 2. Query Current Gaze
    const gaze = this.eyeTracker.currentGaze();
    const gazePxX = gaze.x * w;
    const gazePxY = gaze.y * h;

    // Add to history
    this.gazeTrailHistory.push({ x: gazePxX, y: gazePxY, time: now });
    // Keep last 35 points (~550ms)
    if (this.gazeTrailHistory.length > 35) {
      this.gazeTrailHistory.shift();
    }

    // Log telemetry point if recording
    if (this.isRecording()) {
      this.telemetryLog.push({
        timeMs: Math.round(elapsedSessionMs),
        normX: Number(gaze.x.toFixed(4)),
        normY: Number(gaze.y.toFixed(4)),
        screenX: Math.round(gazePxX),
        screenY: Math.round(gazePxY),
        fixationDurationMs: gaze.fixationDurationMs,
        confidence: gaze.confidence
      });
    }

    // 3. Draw Gaze Trail (Saccade Path)
    if (this.showGazeTrail() && this.gazeTrailHistory.length > 1) {
      ctx.save();
      for (let i = 1; i < this.gazeTrailHistory.length; i++) {
        const pt = this.gazeTrailHistory[i];
        const age = now - pt.time;
        const alpha = Math.max(0, 1 - age / 550);
        ctx.strokeStyle = `rgba(244, 63, 94, ${alpha * 0.45})`;
        ctx.lineWidth = 3 * alpha;
        ctx.beginPath();
        ctx.moveTo(this.gazeTrailHistory[i - 1].x, this.gazeTrailHistory[i - 1].y);
        ctx.lineTo(pt.x, pt.y);
        ctx.stroke();

        // Small node dot
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.7})`;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2.5 * alpha, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // 4. Draw Primary Gaze Reticle
    this.drawReticle(ctx, gazePxX, gazePxY, gaze.fixationDurationMs);

    // 5. Draw Top Picture-in-Picture Webcam Overlay (The Viral YouTube Style)
    this.drawWebcamPip(ctx, w);

    // 6. Draw Telemetry HUD Banner
    if (this.showTelemetryHud()) {
      this.drawTelemetryHud(ctx, w, h, elapsedSessionMs, gaze, gazePxX, gazePxY);
    }
  }

  private drawReticle(ctx: CanvasRenderingContext2D, x: number, y: number, fixationMs: number): void {
    ctx.save();
    const style = this.reticleStyle();
    const baseColor = this.reticleColor();
    const isLocked = fixationMs > 200;

    if (style === 'neon-reticle') {
      // Classic YouTube Gamer Gaze Reticle
      const radius = isLocked ? 28 : 22;

      // Glow circle
      ctx.shadowColor = baseColor;
      ctx.shadowBlur = isLocked ? 25 : 12;
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 3;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();

      // Outer rotating brackets
      const angle = (performance.now() * 0.003) % (Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';

      ctx.beginPath();
      ctx.arc(x, y, radius + 8, angle, angle + Math.PI * 0.35);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, radius + 8, angle + Math.PI, angle + Math.PI * 1.35);
      ctx.stroke();

      // Center crosshair dot
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();

      // Fixation pulse text if locked
      if (isLocked) {
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#fecdd3';
        ctx.font = '700 11px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText(`LOCKED ${fixationMs}ms`, x, y + radius + 20);
      }
    } else if (style === 'laser-pointer') {
      // Sharp high-intensity laser
      ctx.shadowColor = '#ef4444';
      ctx.shadowBlur = 20;
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    } else if (style === 'heat-focus') {
      // Soft radial gradient heat blob
      const gradient = ctx.createRadialGradient(x, y, 4, x, y, 50);
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0.9)');
      gradient.addColorStop(0.3, 'rgba(245, 158, 11, 0.7)');
      gradient.addColorStop(0.7, 'rgba(59, 130, 246, 0.3)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 50, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Bullseye scientific reticle
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      [14, 26, 38].forEach(r => {
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.stroke();
      });

      // Crosshair lines
      ctx.beginPath();
      ctx.moveTo(x - 45, y);
      ctx.lineTo(x + 45, y);
      ctx.moveTo(x, y - 45);
      ctx.lineTo(x, y + 45);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawWebcamPip(ctx: CanvasRenderingContext2D, w: number): void {
    const videoEl = this.eyeTracker.getVideoElement();
    if (!videoEl || videoEl.readyState < 2) return;

    ctx.save();
    // PiP Dimension: 280 x 175 px (16:10 or 16:9 aspect)
    const pipW = 270;
    const pipH = 165;
    let pipX = w - pipW - 24;
    let pipY = 24;

    const pos = this.pipPosition();
    if (pos === 'top-left') {
      pipX = 24;
      pipY = 24;
    } else if (pos === 'top-center') {
      pipX = (w - pipW) / 2;
      pipY = 20;
    }

    // Outer shadow / glow border
    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
    ctx.shadowBlur = 18;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 6;

    // Draw Rounded Rectangle Path
    const radius = 12;
    ctx.beginPath();
    ctx.roundRect(pipX, pipY, pipW, pipH, radius);
    ctx.closePath();
    ctx.fillStyle = '#0f172a';
    ctx.fill();

    // Clip to rounded rectangle to draw video inside
    ctx.save();
    ctx.clip();
    // Mirror webcam feed horizontally so it feels natural to user
    ctx.translate(pipX + pipW, pipY);
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, 0, 0, pipW, pipH);
    ctx.restore();

    // Border around PiP
    ctx.shadowBlur = 0;
    ctx.strokeStyle = this.isRecording() ? '#ef4444' : '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Badge on PiP: "WEBCAM EYE TRACKER"
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(pipX + 8, pipY + 8, 140, 22);
    ctx.fillStyle = this.isRecording() ? '#f87171' : '#38bdf8';
    ctx.beginPath();
    ctx.arc(pipX + 18, pipY + 19, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.fillText('WEBCAM TRACKING', pipX + 28, pipY + 23);

    // Optional Eye Zoom Inset inside PiP corner
    if (this.showEyeCropZoom()) {
      const leftCrop = this.eyeTracker.leftEyeCrop();
      if (leftCrop) {
        // Subtle eye tracking focus badge
        ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
        ctx.fillRect(pipX + pipW - 68, pipY + pipH - 24, 60, 18);
        ctx.fillStyle = '#10b981';
        ctx.font = '700 10px "JetBrains Mono"';
        ctx.fillText('IRIS: OK', pipX + pipW - 62, pipY + pipH - 11);
      }
    }

    ctx.restore();
  }

  private drawTelemetryHud(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    elapsedMs: number,
    gaze: GazePoint,
    pxX: number,
    pxY: number
  ): void {
    ctx.save();
    const hudW = 460;
    const hudH = 34;
    const hudX = 24;
    const hudY = h - hudH - 20;

    // Dark pill container
    ctx.fillStyle = 'rgba(10, 15, 29, 0.88)';
    ctx.beginPath();
    ctx.roundRect(hudX, hudY, hudW, hudH, 8);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Timecode
    const minutes = Math.floor(elapsedMs / 60000);
    const seconds = Math.floor((elapsedMs % 60000) / 1000);
    const ms = Math.floor(elapsedMs % 1000);
    const timeStr = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;

    ctx.fillStyle = this.isRecording() ? '#f43f5e' : '#94a3b8';
    ctx.font = '700 13px "JetBrains Mono", monospace';
    ctx.fillText(this.isRecording() ? '● REC' : '○ IDLE', hudX + 12, hudY + 22);

    ctx.fillStyle = '#f8fafc';
    ctx.fillText(timeStr, hudX + 66, hudY + 22);

    // Gaze Coordinates
    ctx.fillStyle = '#64748b';
    ctx.fillText('·', hudX + 172, hudY + 22);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(`GAZE [${Math.round(pxX)}, ${Math.round(pxY)}]`, hudX + 184, hudY + 22);

    // Fixation
    ctx.fillStyle = '#64748b';
    ctx.fillText('·', hudX + 338, hudY + 22);
    ctx.fillStyle = gaze.fixationDurationMs > 150 ? '#38bdf8' : '#94a3b8';
    ctx.fillText(`FIX ${gaze.fixationDurationMs}ms`, hudX + 350, hudY + 22);

    ctx.restore();
  }

  /**
   * Generates a high-resolution attention heatmap image from recorded gaze telemetry
   */
  generateAttentionHeatmap(): string {
    if (typeof document === 'undefined') return '';
    const heatCanvas = document.createElement('canvas');
    heatCanvas.width = 1280;
    heatCanvas.height = 720;
    const hCtx = heatCanvas.getContext('2d');
    if (!hCtx) return '';

    // Draw dim snapshot of stimulus background
    this.challengeVideos.drawCurrentFrame(hCtx, 1280, 720);
    hCtx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    hCtx.fillRect(0, 0, 1280, 720);

    // Overlay heat blobs for all points in telemetry log
    if (this.telemetryLog.length > 0) {
      // Create radial heatmap points
      for (const pt of this.telemetryLog) {
        const radius = Math.min(80, 35 + (pt.fixationDurationMs / 30));
        const gradient = hCtx.createRadialGradient(pt.screenX, pt.screenY, 2, pt.screenX, pt.screenY, radius);
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.18)');
        gradient.addColorStop(0.4, 'rgba(245, 158, 11, 0.12)');
        gradient.addColorStop(0.8, 'rgba(59, 130, 246, 0.05)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

        hCtx.fillStyle = gradient;
        hCtx.beginPath();
        hCtx.arc(pt.screenX, pt.screenY, radius, 0, Math.PI * 2);
        hCtx.fill();
      }

      // Title watermark
      hCtx.fillStyle = 'rgba(15, 23, 42, 0.9)';
      hCtx.fillRect(24, 24, 380, 64);
      hCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      hCtx.strokeRect(24, 24, 380, 64);

      hCtx.fillStyle = '#f8fafc';
      hCtx.font = '700 16px "Plus Jakarta Sans"';
      hCtx.fillText('GAZE ATTENTION HEATMAP', 40, 50);

      hCtx.fillStyle = '#94a3b8';
      hCtx.font = '500 12px "JetBrains Mono"';
      hCtx.fillText(`TOTAL SAMPLES: ${this.telemetryLog.length} | DURATION: ${(this.recordedDurationMs() / 1000).toFixed(1)}s`, 40, 72);
    }

    const dataUrl = heatCanvas.toDataURL('image/png');
    this.heatmapImageUrl.set(dataUrl);
    return dataUrl;
  }

  /**
   * Exports raw millisecond telemetry data to CSV file
   */
  exportTelemetryCsv(): void {
    if (this.telemetryLog.length === 0) return;

    const headers = 'time_ms,norm_x,norm_y,screen_x,screen_y,fixation_ms,confidence\n';
    const rows = this.telemetryLog
      .map(p => `${p.timeMs},${p.normX},${p.normY},${p.screenX},${p.screenY},${p.fixationDurationMs},${p.confidence}`)
      .join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gazecast_telemetry_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Exports raw telemetry data to JSON
   */
  exportTelemetryJson(): void {
    if (this.telemetryLog.length === 0) return;

    const jsonStr = JSON.stringify({
      sessionTime: new Date().toISOString(),
      durationMs: this.recordedDurationMs(),
      totalDataPoints: this.telemetryLog.length,
      sampleRateHz: 60,
      points: this.telemetryLog
    }, null, 2);

    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gazecast_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async analyzeGazeWithAi(): Promise<void> {
    if (this.telemetryLog.length === 0) return;

    try {
      const summary = {
        totalSamples: this.telemetryLog.length,
        durationSeconds: Math.round(this.recordedDurationMs() / 1000),
        avgFixationDuration: Math.round(
          this.telemetryLog.reduce((acc, p) => acc + p.fixationDurationMs, 0) / this.telemetryLog.length
        ),
        gazePositionsSample: this.telemetryLog.filter((_, i) => i % 15 === 0).map(p => ({
          t: p.timeMs,
          x: p.normX,
          y: p.normY
        }))
      };

      const res = await fetch('/api/ai-analyze-gaze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          challengeName: this.challengeVideos.activePreset().title,
          durationSeconds: summary.durationSeconds,
          telemetrySummary: summary
        })
      });

      if (res.ok) {
        const json = await res.json();
        if (json.data) {
          this.aiInsights.set(json.data);
        }
      }
    } catch (err) {
      console.warn('AI gaze analysis request note:', err);
    }
  }
}
