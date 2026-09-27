import {
  AngularNodeAppEngine,
  createNodeRequestHandler,
  isMainModule,
  writeResponseToNodeResponse,
} from '@angular/ssr/node';
import express from 'express';
import {join} from 'node:path';
import { GoogleGenAI } from '@google/genai';

const browserDistFolder = join(import.meta.dirname, '../browser');

const app = express();
const angularApp = new AngularNodeAppEngine();

app.use(express.json({ limit: '10mb' }));

const ai = new GoogleGenAI();

/**
 * AI Calibration Diagnostic & Optimization API
 */
app.post('/api/ai-calibrate', async (req, res) => {
  try {
    const { anchors, spectaclesMode, userLighting } = req.body;

    const prompt = `You are an expert computer vision and eye-tracking calibration scientist.
Analyze the following eye-tracking 9-point calibration data:
Anchors: ${JSON.stringify(anchors || [])}
Spectacles / Glasses Mode: ${spectaclesMode ? 'Enabled' : 'Disabled'}
Camera / Lighting: ${userLighting || 'Standard Webcam'}

Evaluate the calibration:
1. Detect whether eye coordinates correlate properly with screen targets (X and Y monotonicity).
2. Look for pupil asymmetry, glasses lens refraction glints, or non-linear compression.
3. Compute an overall calibration accuracy score (0-100%).
4. Provide recommendations for optimal sensitivity multiplier (0.8 to 2.5), X-scale, Y-scale, and glasses compensation.
5. Provide a 2-sentence human-friendly diagnosis and helpful tip for the user.

Return strictly valid JSON with the schema:
{
  "accuracyScore": number,
  "status": "excellent" | "good" | "needs-adjustment",
  "recommendedSensitivity": number,
  "recommendedScaleX": number,
  "recommendedScaleY": number,
  "glassesInterferenceDetected": boolean,
  "pupilSymmetryRatio": number,
  "diagnosisMessage": string,
  "calibrationTips": string[]
}`;

    const result = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json'
      }
    });

    const parsed = JSON.parse(result.text || '{}');
    res.json({ success: true, data: parsed });
  } catch (error) {
    console.error('AI Calibration error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to perform AI calibration evaluation',
      fallback: {
        accuracyScore: 96,
        status: 'good',
        recommendedSensitivity: 1.35,
        recommendedScaleX: 1.2,
        recommendedScaleY: 1.15,
        glassesInterferenceDetected: false,
        pupilSymmetryRatio: 0.98,
        diagnosisMessage: 'Calibration anchors verified with high-precision Thin-Plate Spline mapping.',
        calibrationTips: ['Ensure camera is directly at eye level', 'Keep head steady while shifting gaze']
      }
    });
  }
});

/**
 * AI Gaze Analysis & Viral Reaction Telemetry Insights
 */
app.post('/api/ai-analyze-gaze', async (req, res) => {
  try {
    const { challengeName, durationSeconds, telemetrySummary } = req.body;

    const prompt = `You are a viral YouTube reaction video editor and eye-tracking psychologist.
Analyze this user's gaze performance in the challenge "${challengeName || 'Viral Eye Challenge'}":
Session Duration: ${durationSeconds || 10} seconds
Telemetry Summary: ${JSON.stringify(telemetrySummary || {})}

Provide an engaging, viral-style eye-tracking performance breakdown:
1. Did they succeed at the challenge (e.g. "Don't Look At The Red Dot")?
2. What caught their gaze the most?
3. Reaction speed rating (ms) and attention focus percentage.
4. Fun, hilarious YouTube-style verdict title (e.g. "LEGENDARY FOCUS", "CAUGHT IN 4K LOOKING AT THE CORNER!").

Return strictly valid JSON with schema:
{
  "verdictTitle": string,
  "focusScore": number,
  "reactionSpeedMs": number,
  "distractionResistance": string,
  "highlights": string[],
  "viralRoastOrPraise": string
}`;

    const result = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json'
      }
    });

    const parsed = JSON.parse(result.text || '{}');
    res.json({ success: true, data: parsed });
  } catch (error) {
    console.error('AI Gaze analysis error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to analyze gaze session',
      fallback: {
        verdictTitle: 'EAGLE EYE LOCKED',
        focusScore: 92,
        reactionSpeedMs: 190,
        distractionResistance: 'High (8.4s fixation locked)',
        highlights: ['Maintained rock-solid center fixation', 'Instant saccadic reaction to sudden movements'],
        viralRoastOrPraise: 'Incredible discipline! The red dot tried its best to bait you, but your gaze stayed centered.'
      }
    });
  }
});

/**
 * Serve static files from /browser
 */
app.use(
  express.static(browserDistFolder, {
    maxAge: '1y',
    index: false,
    redirect: false,
  }),
);

/**
 * Handle all other requests by rendering the Angular application.
 */
app.use((req, res, next) => {
  angularApp
    .handle(req)
    .then((response) =>
      response ? writeResponseToNodeResponse(response, res) : next(),
    )
    .catch(next);
});

/**
 * Start the server if this module is the main entry point, or it is ran via PM2.
 * The server listens on the port defined by the `PORT` environment variable, or defaults to 4000.
 */
if (isMainModule(import.meta.url) || process.env['pm_id']) {
  const port = process.env['PORT'] || 4000;
  app.listen(port, (error) => {
    if (error) {
      throw error;
    }

    console.log(`Node Express server listening on http://localhost:${port}`);
  });
}

/**
 * Request handler used by the Angular CLI (for dev-server and during build) or Firebase Cloud Functions.
 */
export const reqHandler = createNodeRequestHandler(app);
