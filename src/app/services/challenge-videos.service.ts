import { Injectable, signal } from '@angular/core';

export interface ChallengePreset {
  id: string;
  title: string;
  description: string;
  durationSec: number;
  difficulty: 'Easy' | 'Hard' | 'Extreme';
  category: string;
}

@Injectable({
  providedIn: 'root'
})
export class ChallengeVideosService {
  readonly presets: ChallengePreset[] = [
    {
      id: 'red-dot',
      title: 'Don\'t Look At The Red Circle!',
      description: 'The viral YouTube classic. Try your hardest NOT to look at the moving red target while distractors flash around it.',
      durationSec: 25,
      difficulty: 'Hard',
      category: 'Visual Distraction'
    },
    {
      id: 'cup-shuffle',
      title: '3-Cup Magician Sleight of Hand',
      description: 'A diamond is placed under Cup #2, then shuffled at lightning speed across the table. Can your eyes track it?',
      durationSec: 22,
      difficulty: 'Extreme',
      category: 'Motion Tracking'
    },
    {
      id: 'attention-grid',
      title: 'Where Do You Look First? Test',
      description: 'Sudden high-contrast characters, food, faces, and optical illusions pop onto screen quadrants to measure subconscious bias.',
      durationSec: 20,
      difficulty: 'Easy',
      category: 'Fixation Bias'
    }
  ];

  readonly activePreset = signal<ChallengePreset>(this.presets[0]);
  readonly customVideoUrl = signal<string | null>(null);
  readonly customVideoName = signal<string | null>(null);
  readonly isCustomVideo = signal<boolean>(false);
  readonly isPlaying = signal<boolean>(false);
  readonly currentTime = signal<number>(0);
  readonly duration = signal<number>(25);

  // Challenge renderer canvas
  private challengeCanvas: HTMLCanvasElement | null = null;
  private challengeCtx: CanvasRenderingContext2D | null = null;
  private animId: number | null = null;
  private startTime = 0;
  private audioCtx: AudioContext | null = null;

  // Custom HTML Video Element for uploaded videos
  private customVideoEl: HTMLVideoElement | null = null;

  constructor() {
    if (typeof document !== 'undefined') {
      this.challengeCanvas = document.createElement('canvas');
      this.challengeCanvas.width = 1280;
      this.challengeCanvas.height = 720;
      this.challengeCtx = this.challengeCanvas.getContext('2d');
    }
  }

  getCanvas(): HTMLCanvasElement | null {
    return this.challengeCanvas;
  }

  getCustomVideoElement(): HTMLVideoElement | null {
    return this.customVideoEl;
  }

  selectPreset(preset: ChallengePreset) {
    this.activePreset.set(preset);
    this.isCustomVideo.set(false);
    this.duration.set(preset.durationSec);
    this.currentTime.set(0);
    this.stop();
  }

  loadCustomVideoFile(file: File) {
    if (!this.customVideoEl) {
      this.customVideoEl = document.createElement('video');
      this.customVideoEl.crossOrigin = 'anonymous';
      this.customVideoEl.playsInline = true;
    }

    const url = URL.createObjectURL(file);
    this.customVideoUrl.set(url);
    this.customVideoName.set(file.name);
    this.isCustomVideo.set(true);

    this.customVideoEl.src = url;
    this.customVideoEl.onloadedmetadata = () => {
      if (this.customVideoEl) {
        this.duration.set(Math.round(this.customVideoEl.duration || 30));
        this.currentTime.set(0);
      }
    };
  }

  play() {
    this.isPlaying.set(true);
    if (this.isCustomVideo() && this.customVideoEl) {
      this.customVideoEl.play();
    } else {
      this.startTime = performance.now() - (this.currentTime() * 1000);
      this.startSynthesizerLoop();
    }
  }

  pause() {
    this.isPlaying.set(false);
    if (this.isCustomVideo() && this.customVideoEl) {
      this.customVideoEl.pause();
    }
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  stop() {
    this.pause();
    this.currentTime.set(0);
    if (this.isCustomVideo() && this.customVideoEl) {
      this.customVideoEl.currentTime = 0;
    }
  }

  seek(seconds: number) {
    this.currentTime.set(seconds);
    if (this.isCustomVideo() && this.customVideoEl) {
      this.customVideoEl.currentTime = seconds;
    } else {
      this.startTime = performance.now() - (seconds * 1000);
    }
  }

  /**
   * Draws the active frame of the stimulus (either the custom video or the procedural challenge)
   */
  drawCurrentFrame(targetCtx: CanvasRenderingContext2D, targetWidth: number, targetHeight: number) {
    if (this.isCustomVideo() && this.customVideoEl) {
      if (this.customVideoEl.readyState >= 2) {
        targetCtx.drawImage(this.customVideoEl, 0, 0, targetWidth, targetHeight);
      } else {
        // Fallback placeholder while loading
        targetCtx.fillStyle = '#0f172a';
        targetCtx.fillRect(0, 0, targetWidth, targetHeight);
      }
    } else if (this.challengeCanvas) {
      targetCtx.drawImage(this.challengeCanvas, 0, 0, targetWidth, targetHeight);
    } else {
      targetCtx.fillStyle = '#050811';
      targetCtx.fillRect(0, 0, targetWidth, targetHeight);
    }
  }

  private startSynthesizerLoop() {
    if (this.animId) cancelAnimationFrame(this.animId);

    const render = () => {
      if (!this.isPlaying()) return;

      const elapsed = (performance.now() - this.startTime) / 1000;
      this.currentTime.set(Math.min(this.duration(), elapsed));

      if (elapsed >= this.duration()) {
        this.stop();
        return;
      }

      this.renderProceduralChallenge(this.activePreset().id, elapsed);
      this.animId = requestAnimationFrame(render);
    };

    this.animId = requestAnimationFrame(render);
  }

  /**
   * High-definition synthesized challenge rendering (1280x720 60FPS)
   */
  private renderProceduralChallenge(challengeId: string, t: number) {
    if (!this.challengeCtx || !this.challengeCanvas) return;
    const ctx = this.challengeCtx;
    const w = this.challengeCanvas.width;
    const h = this.challengeCanvas.height;

    // Dark sleek backdrop with subtle grid
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, w, h);

    // Subtle background grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 80) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 80) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (challengeId === 'red-dot') {
      this.renderRedDotChallenge(ctx, w, h, t);
    } else if (challengeId === 'cup-shuffle') {
      this.renderCupShuffleChallenge(ctx, w, h, t);
    } else {
      this.renderAttentionGridChallenge(ctx, w, h, t);
    }

    // Top Header Watermark
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, 0, w, 54);
    ctx.fillStyle = '#f8fafc';
    ctx.font = '600 20px "Plus Jakarta Sans", sans-serif';
    ctx.fillText(`CHALLENGE: ${this.activePreset().title.toUpperCase()}`, 30, 34);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '500 15px "JetBrains Mono", monospace';
    const remain = Math.max(0, this.duration() - t).toFixed(1);
    ctx.fillText(`REMAINING: ${remain}s`, w - 180, 34);
    ctx.restore();
  }

  private renderRedDotChallenge(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
    // Stage warning banner
    ctx.save();
    ctx.textAlign = 'center';

    if (t < 3) {
      ctx.fillStyle = '#f43f5e';
      ctx.font = '700 48px "Plus Jakarta Sans", sans-serif';
      ctx.fillText('DO NOT LOOK AT THE RED DOT!', w / 2, h / 2 - 30);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '500 20px "Plus Jakarta Sans", sans-serif';
      ctx.fillText('Look anywhere else on the screen. Target begins in ' + (3 - t).toFixed(1) + 's', w / 2, h / 2 + 25);
    } else {
      // The tricky red dot moves along a smooth lissajous path + sudden bursts
      const animT = t - 3;
      const speed = 1.2 + Math.min(2.5, animT * 0.1);
      const dotX = w / 2 + Math.sin(animT * speed) * (w * 0.35) + Math.cos(animT * 2.3) * (w * 0.08);
      const dotY = h / 2 + Math.cos(animT * speed * 0.8) * (h * 0.32) + Math.sin(animT * 1.7) * (h * 0.06);

      // Distractor 1: Flashing green star
      const distractorX = w / 2 + Math.sin(animT * -1.8) * (w * 0.3);
      const distractorY = h / 2 + Math.cos(animT * -1.4) * (h * 0.28);
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(distractorX, distractorY, 26, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 13px "JetBrains Mono"';
      ctx.fillText('SAFE ZONE', distractorX, distractorY + 4);

      // Distractor 2: Sudden flashing text cards
      if (Math.sin(animT * 3) > 0.4) {
        ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
        ctx.fillRect(w * 0.15, h * 0.72, 280, 54);
        ctx.fillStyle = '#ffffff';
        ctx.font = '600 18px "Plus Jakarta Sans"';
        ctx.fillText('🚨 POPUP DISTRACTION!', w * 0.15 + 140, h * 0.72 + 34);
      }

      // Distractor 3: Spinning hypnotic rings on opposite side
      ctx.strokeStyle = 'rgba(234, 179, 8, 0.4)';
      ctx.lineWidth = 4;
      const spinAngle = animT * 4;
      ctx.beginPath();
      ctx.arc(w * 0.82, h * 0.28, 45, spinAngle, spinAngle + Math.PI * 1.5);
      ctx.stroke();

      // THE BANNED RED DOT (glow + pulsing ring)
      ctx.shadowColor = '#f43f5e';
      ctx.shadowBlur = 25;
      ctx.fillStyle = '#e11d48';
      ctx.beginPath();
      const dotRadius = 32 + Math.sin(animT * 8) * 6;
      ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
      ctx.fill();

      // Inner white center
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(dotX, dotY, 10, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = '#ffe4e6';
      ctx.font = '700 12px "JetBrains Mono"';
      ctx.fillText('DON\'T LOOK!', dotX, dotY + dotRadius + 20);
    }
    ctx.restore();
  }

  private renderCupShuffleChallenge(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
    ctx.save();
    ctx.textAlign = 'center';

    // Wooden table background
    ctx.fillStyle = '#1e1b18';
    ctx.fillRect(0, h * 0.45, w, h * 0.55);
    ctx.strokeStyle = '#2d2720';
    ctx.lineWidth = 3;
    ctx.strokeRect(0, h * 0.45, w, h * 0.55);

    // Cup positions
    const baseY = h * 0.62;
    const cupSpacing = w * 0.24;
    const centerOffset = w * 0.5;

    // Cup centers
    let c1 = centerOffset - cupSpacing;
    let c2 = centerOffset;
    let c3 = centerOffset + cupSpacing;

    if (t < 3) {
      // Intro: show diamond under cup #2
      ctx.fillStyle = '#f8fafc';
      ctx.font = '700 36px "Plus Jakarta Sans"';
      ctx.fillText('KEEP YOUR EYES ON CUP #2!', w / 2, h * 0.25);

      // Lift Cup #2
      this.drawCup(ctx, c1, baseY, '1');
      this.drawCup(ctx, c2, baseY - 80, '2');
      this.drawCup(ctx, c3, baseY, '3');

      // The Diamond Prize
      ctx.fillStyle = '#38bdf8';
      ctx.font = '40px sans-serif';
      ctx.fillText('💎', c2, baseY + 10);
    } else {
      // Shuffling animation
      const shuffleT = t - 3;
      const round = Math.floor(shuffleT * 1.5);
      const frac = (shuffleT * 1.5) % 1;
      const smoothFrac = 0.5 - 0.5 * Math.cos(frac * Math.PI);

      // Permutations based on round
      if (round % 3 === 0) {
        // Swap 1 and 2
        c1 = (centerOffset - cupSpacing) + smoothFrac * cupSpacing;
        c2 = centerOffset - smoothFrac * cupSpacing;
      } else if (round % 3 === 1) {
        // Swap 2 and 3
        c2 = centerOffset + smoothFrac * cupSpacing;
        c3 = (centerOffset + cupSpacing) - smoothFrac * cupSpacing;
      } else {
        // Swap 1 and 3
        c1 = (centerOffset - cupSpacing) + smoothFrac * (cupSpacing * 2);
        c3 = (centerOffset + cupSpacing) - smoothFrac * (cupSpacing * 2);
      }

      ctx.fillStyle = '#e2e8f0';
      ctx.font = '600 24px "Plus Jakarta Sans"';
      ctx.fillText('FAST SHUFFLE! WHERE IS THE DIAMOND?', w / 2, h * 0.2);

      this.drawCup(ctx, c1, baseY, '?');
      this.drawCup(ctx, c2, baseY, '?');
      this.drawCup(ctx, c3, baseY, '?');
    }

    ctx.restore();
  }

  private drawCup(ctx: CanvasRenderingContext2D, x: number, y: number, label: string) {
    ctx.save();
    ctx.fillStyle = '#d97706';
    ctx.beginPath();
    ctx.moveTo(x - 55, y + 45);
    ctx.lineTo(x + 55, y + 45);
    ctx.lineTo(x + 40, y - 55);
    ctx.lineTo(x - 40, y - 55);
    ctx.closePath();
    ctx.fill();

    // Metallic highlight
    ctx.fillStyle = 'rgba(253, 230, 138, 0.4)';
    ctx.beginPath();
    ctx.moveTo(x - 20, y + 45);
    ctx.lineTo(x, y + 45);
    ctx.lineTo(x - 5, y - 55);
    ctx.lineTo(x - 22, y - 55);
    ctx.closePath();
    ctx.fill();

    // Cup label
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 22px "JetBrains Mono"';
    ctx.fillText(label, x, y + 5);
    ctx.restore();
  }

  private renderAttentionGridChallenge(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
    ctx.save();
    ctx.textAlign = 'center';

    // 4 Quadrants
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const phase = Math.floor(t / 4) % 4;

    // Quadrant centers
    const q1 = { x: w * 0.25, y: h * 0.32, label: 'Quadrant A: Human Face' };
    const q2 = { x: w * 0.75, y: h * 0.32, label: 'Quadrant B: Delicious Burger' };
    const q3 = { x: w * 0.25, y: h * 0.75, label: 'Quadrant C: Glowing Gold Coin' };
    const q4 = { x: w * 0.75, y: h * 0.75, label: 'Quadrant D: Dangerous Creature' };

    ctx.fillStyle = '#f8fafc';
    ctx.font = '700 28px "Plus Jakarta Sans"';
    ctx.fillText('WHICH QUADRANT DOES YOUR BRAIN LOOK AT FIRST?', w / 2, 45);

    // Q1: Face
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(q1.x, q1.y, 60, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = '50px sans-serif';
    ctx.fillText('👤', q1.x, q1.y + 16);

    // Q2: Food
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(q2.x, q2.y, 60, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = '50px sans-serif';
    ctx.fillText('🍔', q2.x, q2.y + 16);

    // Q3: Money
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(q3.x, q3.y, 60, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = '50px sans-serif';
    ctx.fillText('💰', q3.x, q3.y + 16);

    // Q4: Monster
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(q4.x, q4.y, 60, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = '50px sans-serif';
    ctx.fillText('🦖', q4.x, q4.y + 16);

    // Flash a random spotlight on one quadrant
    const targets = [q1, q2, q3, q4];
    const highlighted = targets[phase];
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.strokeRect(highlighted.x - 90, highlighted.y - 85, 180, 170);

    ctx.restore();
  }
}
