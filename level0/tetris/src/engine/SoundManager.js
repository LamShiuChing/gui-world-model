export default class SoundManager {
    constructor() {
        this.ctx = null;
        this.masterGain = null;
        this.volume = 1.0;
        this.muted = false;
        this.noiseBuffer = null;
    }

    init() {
        if (!this.ctx) {
            try {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (AudioContext) {
                    this.ctx = new AudioContext();
                    this.masterGain = this.ctx.createGain();
                    this.masterGain.gain.value = this.volume;
                    this.masterGain.connect(this.ctx.destination);
                    this.noiseBuffer = this.createNoiseBuffer();
                }
            } catch (e) {
                console.warn('AudioContext not supported or failed to initialize', e);
            }
        }
    }

    setVolume(v) {
        this.volume = Math.max(0, Math.min(1, v));
        if (this.masterGain) this.masterGain.gain.value = this.volume;
    }

    createNoiseBuffer() {
        if (!this.ctx) return null;
        const bufferSize = this.ctx.sampleRate * 2; // 2 seconds of noise
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            data[i] = Math.random() * 2 - 1;
        }
        return buffer;
    }

    playHardDrop() {
        if (this.muted) return;
        this.init();
        if (!this.ctx) return;
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const t = this.ctx.currentTime;

        // 1. Low frequency "thud" (Sine kick)
        const osc = this.ctx.createOscillator();
        const oscGain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(150, t);
        osc.frequency.exponentialRampToValueAtTime(0.01, t + 0.15);
        oscGain.gain.setValueAtTime(0.5, t);
        oscGain.gain.exponentialRampToValueAtTime(0.01, t + 0.15);
        osc.connect(oscGain);
        oscGain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + 0.15);

        // 2. Impact noise (Filtered noise burst)
        const noise = this.ctx.createBufferSource();
        noise.buffer = this.noiseBuffer;
        const noiseFilter = this.ctx.createBiquadFilter();
        const noiseGain = this.ctx.createGain();

        noiseFilter.type = 'lowpass';
        noiseFilter.frequency.setValueAtTime(800, t);
        noiseFilter.frequency.exponentialRampToValueAtTime(100, t + 0.1);

        noiseGain.gain.setValueAtTime(0.3, t);
        noiseGain.gain.exponentialRampToValueAtTime(0.01, t + 0.1);

        noise.connect(noiseFilter);
        noiseFilter.connect(noiseGain);
        noiseGain.connect(this.masterGain);
        noise.start(t);
        noise.stop(t + 0.1);
    }

    playLineClear() {
        if (this.muted) return;
        this.init();
        if (!this.ctx) return;
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const t = this.ctx.currentTime;

        // Pure Bell: Single Sine Wave at A5 (880Hz)
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, t);

        gain.gain.setValueAtTime(0, t);
        gain.gain.linearRampToValueAtTime(0.3, t + 0.01); // Instant attack
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.8); // Medium release

        osc.connect(gain);
        gain.connect(this.masterGain);

        osc.start(t);
        osc.stop(t + 0.8);
    }

    playTSpin() {
        if (this.muted) return;
        this.init();
        if (!this.ctx) return;
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const t = this.ctx.currentTime;

        // Heavy Cog (FM Synthesis + Grinding Noise)

        // 1. FM Synthesis for Metallic "Clank/Grind"
        const carrier = this.ctx.createOscillator();
        const carrierGain = this.ctx.createGain();
        const modulator = this.ctx.createOscillator();
        const modulatorGain = this.ctx.createGain();

        // Carrier: Low frequency for "heavy" weight
        carrier.type = 'sine';
        carrier.frequency.setValueAtTime(60, t); // Deep base

        // Modulator: Inharmonic ratio for metallic texture
        modulator.type = 'square';
        modulator.frequency.setValueAtTime(144, t); // 2.4x ratio

        // Modulation Envelope
        modulatorGain.gain.setValueAtTime(0, t);
        modulatorGain.gain.linearRampToValueAtTime(300, t + 0.05); // Heavy modulation
        modulatorGain.gain.exponentialRampToValueAtTime(10, t + 0.3);

        // Amplitude Envelope
        carrierGain.gain.setValueAtTime(0, t);
        carrierGain.gain.linearRampToValueAtTime(0.4, t + 0.05); // Slower attack for mass
        carrierGain.gain.exponentialRampToValueAtTime(0.01, t + 0.4);

        // FM Connections
        modulator.connect(modulatorGain);
        modulatorGain.connect(carrier.frequency);
        carrier.connect(carrierGain);
        carrierGain.connect(this.masterGain);

        // 2. Grinding Noise Layer
        const noise = this.ctx.createBufferSource();
        noise.buffer = this.noiseBuffer;
        const noiseFilter = this.ctx.createBiquadFilter();
        const noiseGain = this.ctx.createGain();

        noiseFilter.type = 'lowpass';
        noiseFilter.frequency.setValueAtTime(200, t); // Low rumble

        noiseGain.gain.setValueAtTime(0, t);
        noiseGain.gain.linearRampToValueAtTime(0.2, t + 0.05);
        noiseGain.gain.linearRampToValueAtTime(0, t + 0.3);

        noise.connect(noiseFilter);
        noiseFilter.connect(noiseGain);
        noiseGain.connect(this.masterGain);

        // Start/Stop
        carrier.start(t);
        modulator.start(t);
        noise.start(t);

        carrier.stop(t + 0.4);
        modulator.stop(t + 0.4);
        noise.stop(t + 0.4);
    }

    playLevelUp() {
        if (this.muted) return;
        this.init();
        if (!this.ctx) return;
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const t = this.ctx.currentTime;

        // Two-Note Ascending (C4 -> G4)
        // Note 1: C4 (261.63 Hz)
        const osc1 = this.ctx.createOscillator();
        const gain1 = this.ctx.createGain();
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(261.63, t);

        gain1.gain.setValueAtTime(0, t);
        gain1.gain.linearRampToValueAtTime(0.3, t + 0.05);
        gain1.gain.exponentialRampToValueAtTime(0.01, t + 0.3);

        osc1.connect(gain1);
        gain1.connect(this.masterGain);
        osc1.start(t);
        osc1.stop(t + 0.3);

        // Note 2: G4 (392.00 Hz) - delayed by 150ms
        const t2 = t + 0.15;
        const osc2 = this.ctx.createOscillator();
        const gain2 = this.ctx.createGain();
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(392.00, t2);

        gain2.gain.setValueAtTime(0, t2);
        gain2.gain.linearRampToValueAtTime(0.3, t2 + 0.05);
        gain2.gain.exponentialRampToValueAtTime(0.01, t2 + 0.4);

        osc2.connect(gain2);
        gain2.connect(this.masterGain);
        osc2.start(t2);
        osc2.stop(t2 + 0.4);
    }

    playClick() {
        if (this.muted) return;
        this.init();
        if (!this.ctx) return;

        try {
            if (this.ctx.state === 'suspended') this.ctx.resume();

            const t = this.ctx.currentTime;

            // High pitched "tick"
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();

            osc.type = 'sine';
            osc.frequency.setValueAtTime(800, t);
            osc.frequency.exponentialRampToValueAtTime(1200, t + 0.05);

            gain.gain.setValueAtTime(0.1, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.05);

            osc.connect(gain);
            gain.connect(this.masterGain);

            osc.start(t);
            osc.stop(t + 0.05);
        } catch (e) {
            console.warn('Error playing click sound', e);
        }
    }

    playLock() {
        if (this.muted) return;
        this.init();
        if (!this.ctx) return;
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const t = this.ctx.currentTime;

        // Light "Thud" / "Click" for natural lock
        // Sine wave with quick pitch drop
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(250, t);
        osc.frequency.exponentialRampToValueAtTime(50, t + 0.1);

        gain.gain.setValueAtTime(0.2, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.1);

        osc.connect(gain);
        gain.connect(this.masterGain);

        osc.start(t);
        osc.stop(t + 0.1);
    }

    toggleMute() {
        this.muted = !this.muted;
    }
}
