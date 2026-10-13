import Renderer from './Renderer.js';
import Input from './Input.js';
import Tetromino, { TETROMINOES, KICK_TABLE } from './Tetromino.js';
import SoundManager from './SoundManager.js';

export default class Game {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.renderer = new Renderer(this.ctx);
        this.input = new Input(this);
        this.soundManager = new SoundManager();

        this.lastTime = 0;
        this.dropCounter = 0;
        this.dropInterval = 100; // Very fast master mode starting speed

        // Timer tracking
        this.startTime = 0;
        this.elapsedTime = 0;

        this.lockDelay = 500;
        this.lockCounter = 0;
        this.isLocking = false;
        this.lockMoveResetCount = 0;
        this.MAX_LOCK_RESETS = 15;

        // Board: 10 columns x 40 rows (20 hidden + 20 visible)
        // y=0 is top (hidden), y=19 is bottom of hidden, y=20 is top of visible, y=39 is bottom
        this.BOARD_WIDTH = 10;
        this.BOARD_HEIGHT = 40;
        this.VISIBLE_HEIGHT = 20;
        this.board = this.createBoard();

        this.player = {
            piece: null,
            pos: { x: 0, y: 0 },
            score: 0,
            level: 1,
            lines: 0,
            combo: -1,
            backToBack: false,
            lastAction: null, // 'move', 'rotate', 'drop'
            holdPiece: null, // Held piece type
            canHold: true // Can hold this turn
        };

        this.isRunning = false;
        this.isPaused = false;
        this.gameOver = false;
        this._ihsActive = false; // Recursion guard for IHS

        // 7-Bag Randomizer
        this.bag = [];

        // Effects
        this.particles = [];
        this.shakeTimer = 0;
        this.maxShakeTimer = 0;
        this.shakeIntensity = 0;
        this.shakeMultiplier = 1.0;
        this.bloomTimer = 0;
        this.bloomIntensity = 0;

        // Pause time tracking
        this.totalPausedTime = 0;
        this.pauseStartTime = 0;

        // Popups
        this.popups = [];

        // Line Clear Animation
        this.clearingRows = [];
        this.clearAnimationTimer = 0;
        this.CLEAR_ANIMATION_DURATION = 300;

        // Countdown
        this.isCountingDown = false;
        this.countdownStep = 0;   // 3, 2, 1, 0 (0 = GO)
        this.countdownTimer = 0;
        this.COUNTDOWN_STEP_DURATION = 900;
    }

    createBoard() {
        return Array.from({ length: this.BOARD_HEIGHT }, () => Array(this.BOARD_WIDTH).fill(0));
    }

    start(startLevel = 1) {
        if (this.isRunning) return;
        this.isRunning = true;
        this.gameOver = false;
        this.lastTime = performance.now();
        this.startTime = performance.now();
        this.reset(startLevel);
        this.update(this.lastTime);
    }

    startCountdown(startLevel = 1) {
        if (this.isRunning) return;
        this.isCountingDown = true;
        this.countdownStep = 3;
        this.countdownTimer = 0;
        this.gameOver = false;
        this.isRunning = true;
        this.reset(startLevel);
        // Clear piece/bag so nothing shows during countdown
        this.player.piece = null;
        this.player.holdPiece = null;
        this.bag = [];
        this.lastTime = performance.now();
        this.update(this.lastTime);
    }

    reset(startLevel = 1) {
        this.board = this.createBoard();
        this.startLevel = startLevel;
        this.player.score = 0;
        this.player.level = startLevel;
        this.player.lines = 0; // Level lines
        this.player.totalLines = 0; // Total lines cleared
        this.player.combo = -1;
        this.player.backToBack = false;
        this.player.holdPiece = null;
        this.player.canHold = true;
        this.updateGravity();
        this.bag = []; // Reset bag on game reset
        this.elapsedTime = 0; // Reset timer
        this.totalPausedTime = 0;
        this.pauseStartTime = 0;
        this.spawnPiece();
        this.updateScore();
        this.updateTimer();
        this.clearingRows = [];
        this.clearAnimationTimer = 0;
    }

    spawnPiece() {
        // Ensure bag has enough pieces (maintain at least 7 pieces for preview)
        if (this.bag.length < 7) {
            this.refillBag();
        }
        const type = this.bag.shift();
        this.player.piece = new Tetromino(type);

        // Spawn rules: orientation 0, x centered, y just above visible area
        this.player.pos = { x: 4, y: 18 };
        this.player.piece.orientation = 0;
        this.isLocking = false;
        this.lockCounter = 0;
        this.lockMoveResetCount = 0;
        this.player.lastAction = 'spawn';
        this.player.canHold = true; // Allow hold for new piece

        // Game Over check
        if (this.collide(this.player.pos.x, this.player.pos.y, this.player.piece.orientation)) {
            this.gameOver = true;
            this.isRunning = false;
            document.getElementById('game-over-overlay').classList.remove('hidden');
            document.getElementById('final-score').innerText = `SCORE: ${Math.floor(this.player.score)}`;
            document.getElementById('highscore-form').classList.remove('hidden');
            document.getElementById('player-name').focus();
            return;
        }

        // IHS (Initial Hold System): if hold is buffered, swap immediately on entry
        if (this.input.keys.hold && this.player.canHold && !this._ihsActive) {
            const hadHoldPiece = this.player.holdPiece !== null;
            this._ihsActive = true;
            this.playerHold();
            this._ihsActive = false;
            if (this.gameOver) return;
            // If hold was empty, playerHold called spawnPiece which already applied IRS
            if (!hadHoldPiece) return;
            // Fall through to apply IRS to the swapped-in piece
        }

        // IRS (Initial Rotation System): if a rotate key is buffered, apply on entry
        if (this.player.piece && this.player.piece.type !== 'O') {
            const irsDir = this.input.keys.rotateCCW ? -1 : (this.input.keys.rotateCW ? 1 : 0);
            if (irsDir !== 0) this.playerRotate(irsDir);
        }
    }

    collide(x, y, orientation) {
        const blocks = this.player.piece.getBlocks(x, y, orientation);
        for (const block of blocks) {
            if (block.x < 0 || block.x >= this.BOARD_WIDTH || block.y >= this.BOARD_HEIGHT) {
                return true;
            }
            if (block.y >= 0 && this.board[block.y][block.x] !== 0) {
                return true;
            }
        }
        return false;
    }

    playerMove(dir) {
        if (this.gameOver || !this.player.piece) return;
        const newX = this.player.pos.x + dir;
        if (!this.collide(newX, this.player.pos.y, this.player.piece.orientation)) {
            this.player.pos.x = newX;
            this.player.lastAction = 'move';

            // Check if piece is grounded at new position
            const isGrounded = this.collide(newX, this.player.pos.y + 1, this.player.piece.orientation);

            if (!isGrounded) {
                // Moved off a platform into air - cancel locking
                this.isLocking = false;
                this.lockCounter = 0;
            } else if (this.isLocking && this.lockMoveResetCount < this.MAX_LOCK_RESETS) {
                // Still grounded - reset lock timer
                this.lockCounter = 0;
                this.lockMoveResetCount++;
            }
        }
    }

    playerRotate(dir) {
        if (this.gameOver || !this.player.piece) return;
        const piece = this.player.piece;
        const currentOrient = piece.orientation;
        const newOrient = (currentOrient + dir + 4) % 4;

        // O-piece: no rotation
        if (piece.type === 'O') return;

        const kickKey = `${currentOrient}-${newOrient}`;
        const table = piece.type === 'I' ? KICK_TABLE.I : KICK_TABLE.JLSTZ;
        const kicks = table[kickKey];

        for (const [dx, dy] of kicks) {
            const testX = this.player.pos.x + dx;
            const testY = this.player.pos.y + dy;

            if (!this.collide(testX, testY, newOrient)) {
                this.player.pos.x = testX;
                this.player.pos.y = testY;
                piece.orientation = newOrient;
                this.player.lastAction = 'rotate';

                // Check for T-Spin sound
                if (this.isTSpin()) {
                    this.soundManager.playTSpin();
                }

                // Check if piece is grounded after rotation
                const isGrounded = this.collide(testX, testY + 1, newOrient);

                if (isGrounded) {
                    // Piece is on the ground, start/reset locking
                    if (this.lockMoveResetCount < this.MAX_LOCK_RESETS) {
                        this.isLocking = true;
                        this.lockCounter = 0;
                        this.lockMoveResetCount++;
                    }
                } else {
                    // Piece is in mid-air, cancel locking
                    this.isLocking = false;
                    this.lockCounter = 0;
                }

                return;
            }
        }
    }

    playerDrop(soft = false) {
        if (this.gameOver || !this.player.piece) return;
        const newY = this.player.pos.y + 1;
        if (!this.collide(this.player.pos.x, newY, this.player.piece.orientation)) {
            this.player.pos.y = newY;
            this.player.lastAction = 'drop';
            if (soft) {
                this.player.score += 1;
                this.updateScore();
            }
            this.isLocking = false;
            this.lockCounter = 0;
            return true; // Moved
        } else {
            // Grounded
            this.isLocking = true;
            return false; // Didn't move
        }
    }

    playerHardDrop() {
        if (this.gameOver || !this.player.piece) return;
        let droppedCells = 0;
        while (!this.collide(this.player.pos.x, this.player.pos.y + 1, this.player.piece.orientation)) {
            this.player.pos.y++;
            droppedCells++;
        }
        this.player.score += droppedCells * 2;
        this.player.lastAction = 'drop';
        this.soundManager.playHardDrop();
        this.lock(true); // Pass true to indicate hard drop
        this.updateScore();
    }

    update(time = 0) {
        if (!this.isRunning) return;

        if (this.isPaused) {
            this.lastTime = time;
            requestAnimationFrame(this.update.bind(this));
            return;
        }

        const deltaTime = time - this.lastTime;
        this.lastTime = time;

        // Countdown phase — render board but skip all physics/input
        if (this.isCountingDown) {
            this.countdownTimer += deltaTime;
            if (this.countdownTimer >= this.COUNTDOWN_STEP_DURATION) {
                this.countdownTimer -= this.COUNTDOWN_STEP_DURATION;
                this.countdownStep--;
                if (this.countdownStep < 0) {
                    this.isCountingDown = false;
                    this.startTime = performance.now();
                    this.elapsedTime = 0;
                    this.totalPausedTime = 0;
                    this.spawnPiece();
                }
            }
            this.renderer.draw(this.board, this.player, this, { x: 0, y: 0 }, [], 0, []);
            requestAnimationFrame(this.update.bind(this));
            return;
        }

        // Update timer
        this.elapsedTime = time - this.startTime - this.totalPausedTime;
        this.updateTimer();

        // Handle Line Clear Animation
        if (this.clearAnimationTimer > 0) {
            this.clearAnimationTimer -= deltaTime;
            if (this.clearAnimationTimer <= 0) {
                this.finalizeClear();
            }
        }

        // Input & Gravity - Only if not animating
        if (this.clearAnimationTimer <= 0) {
            // Update input (DAS/ARR)
            this.input.update(deltaTime);

            // Gravity Handling
            if (!this.isLocking) {
                if (this.dropInterval === 0) {
                    // 20G Mode: Instant Drop
                    // In 20G, we try to drop as far as possible every frame
                    // But we don't lock immediately, we just sit on the stack
                    while (this.playerDrop()) {
                        // Keep dropping until grounded
                    }
                    // Once grounded, locking starts naturally via playerDrop returning false
                } else {
                    // Standard Gravity
                    this.dropCounter += deltaTime;
                    if (this.dropCounter > this.dropInterval) {
                        this.playerDrop();
                        this.dropCounter = 0;
                    }
                }
            } else {
                this.lockCounter += deltaTime;
                if (this.lockCounter > this.lockDelay) {
                    this.lock();
                }
            }
        }

        // Update Shake
        if (this.shakeTimer > 0) {
            this.shakeTimer -= deltaTime;
            if (this.shakeTimer < 0) this.shakeTimer = 0;
        }

        // Update Particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.05; // Slightly heavier gravity for sparks
            p.life -= p.decay; // Individual decay rates
            if (p.life <= 0) {
                this.particles.splice(i, 1);
            }
        }

        // Update Popups
        for (let i = this.popups.length - 1; i >= 0; i--) {
            const p = this.popups[i];
            p.y += p.vy;
            p.life -= 0.012;

            if (p.life <= 0) {
                this.popups.splice(i, 1);
            }
        }

        // Calculate shake offset
        let shakeOffset = { x: 0, y: 0 };
        if (this.shakeTimer > 0) {
            // Single impulse recoil: Start at max offset and slide back to 0
            const progress = this.shakeTimer / this.maxShakeTimer;
            // Use a power curve for "heavy" feel (stays down a bit then slides up)
            const magnitude = this.shakeIntensity * Math.pow(progress, 2);

            // Vertical only, single downward displacement
            shakeOffset.x = 0;
            shakeOffset.y = magnitude;
        }

        // Update Bloom
        if (this.bloomTimer > 0) {
            this.bloomTimer -= deltaTime;
            if (this.bloomTimer < 0) this.bloomTimer = 0;
            this.bloomIntensity = 30 * (this.bloomTimer / 300);
        } else {
            this.bloomIntensity = 0;
        }

        this.renderer.draw(this.board, this.player, this, shakeOffset, this.particles, this.bloomIntensity, this.popups);
        requestAnimationFrame(this.update.bind(this));
    }

    lock(isHardDrop = false) {
        // Play lock sound if not a hard drop (hard drop has its own sound)
        if (!isHardDrop) {
            this.soundManager.playLock();
        }

        // Lock piece into board
        const blocks = this.player.piece.getBlocks(this.player.pos.x, this.player.pos.y, this.player.piece.orientation);
        for (const block of blocks) {
            if (block.y >= 0 && block.y < this.BOARD_HEIGHT) {
                this.board[block.y][block.x] = this.player.piece.color;
            }
        }

        const isClearing = this.clearLines();

        this.player.piece = null; // Disable control

        if (isClearing) {
            return; // Wait for animation
        }

        // If no line clear shake was triggered, trigger a landing shake
        if (this.shakeTimer === 0) {
            // Random duration: 250ms - 400ms
            const duration = 250 + Math.random() * 150;
            this.shakeTimer = duration;
            this.maxShakeTimer = duration;

            // Random intensity: 6 - 12
            this.shakeIntensity = (6 + Math.random() * 6) * this.shakeMultiplier;
        }

        // Check Victory Condition
        if (this.player.totalLines >= 300) {
            this.victory();
            return;
        }

        this.spawnPiece();
    }

    clearLines() {
        let linesCleared = 0;
        const tSpinType = this.isTSpin();

        // Sweep lines
        const linesToClear = [];
        for (let y = 0; y < this.BOARD_HEIGHT; y++) {
            let rowFull = true;
            for (let x = 0; x < this.BOARD_WIDTH; x++) {
                if (this.board[y][x] === 0) {
                    rowFull = false;
                    break;
                }
            }
            if (rowFull) {
                linesToClear.push(y);
            }
        }

        if (linesToClear.length > 0) {
            linesCleared = linesToClear.length;

            // Trigger Shake
            // Random duration: 350ms - 550ms
            const duration = 350 + Math.random() * 200;
            this.shakeTimer = duration;
            this.maxShakeTimer = duration;

            // Random intensity based on lines: Base + 0-5 random
            this.shakeIntensity = ((4 * linesCleared) + Math.random() * 5) * this.shakeMultiplier;

            // Trigger Bloom
            this.bloomTimer = 300;


            // Defer line removal
            if (this.dropInterval === 0 || !this.renderer.showLineClearEffect) {
                // Instant Clear: 20G mode or flash disabled
                const newBoard = this.board.filter((row, index) => !linesToClear.includes(index));
                while (newBoard.length < this.BOARD_HEIGHT) {
                    newBoard.unshift(Array(this.BOARD_WIDTH).fill(0));
                }
                this.board = newBoard;
            } else {
                // Standard Mode: Animation
                this.clearingRows = linesToClear;
                this.clearAnimationTimer = this.CLEAR_ANIMATION_DURATION;
            }
        }

        if (linesCleared > 0 || tSpinType) {
            if (linesCleared > 0) {
                this.soundManager.playLineClear();
            }
            const wasB2B = this.player.backToBack;
            this.calculateScore(linesCleared, tSpinType);

            const pX = 475;
            const pY = 400;

            if (tSpinType === 'full') {
                if (linesCleared === 0) this.addPopup("Twist_Single", pX, pY - 50);
                else if (linesCleared === 1) this.addPopup("Twist_Single", pX, pY - 50);
                else if (linesCleared === 2) this.addPopup("Twist_Double", pX, pY - 50);
                else if (linesCleared === 3) this.addPopup("Twist_Triple", pX, pY - 50);
            } else if (tSpinType === 'mini') {
                this.addPopup("T-MINI", pX, pY - 50);
            }

            if (linesCleared === 1 && !tSpinType) this.addPopup("Single", pX, pY);
            else if (linesCleared === 2 && !tSpinType) this.addPopup("Double", pX, pY);
            else if (linesCleared === 3 && !tSpinType) this.addPopup("Triple", pX, pY);
            else if (linesCleared === 4) this.addPopup("Quad", pX, pY);

            if (this.player.combo > 0) {
                this.addPopup(`combo x${this.player.combo}`, pX, pY + 50);
            }

            if (wasB2B && (linesCleared === 4 || tSpinType === 'full')) {
                this.addPopup("B2B", pX, pY + 100);
            }

        } else {
            this.player.combo = -1;
        }

        // Return true if animation is active (standard mode with flash on), false if instant
        return linesCleared > 0 && this.dropInterval !== 0 && this.renderer.showLineClearEffect;
    }

    finalizeClear() {
        // Remove lines and add new ones
        const newBoard = this.board.filter((row, index) => !this.clearingRows.includes(index));
        while (newBoard.length < this.BOARD_HEIGHT) {
            newBoard.unshift(Array(this.BOARD_WIDTH).fill(0));
        }
        this.board = newBoard;
        this.clearingRows = [];

        // Check Victory Condition
        if (this.player.totalLines >= 300) {
            this.victory();
            return;
        }

        this.spawnPiece();
    }

    addPopup(text, x, y) {
        this.popups.push({
            text,
            x,
            y,
            life: 1.0,
            vy: -0.6
        });
    }

    calculateScore(lines, tSpinType) {
        let baseScore = 0;
        let isB2B = false;

        if (tSpinType === 'full') {
            if (lines === 0) { baseScore = 400; }
            else if (lines === 1) { baseScore = 800;  isB2B = true; }
            else if (lines === 2) { baseScore = 1200; isB2B = true; }
            else if (lines === 3) { baseScore = 1600; isB2B = true; }
        } else if (tSpinType === 'mini') {
            if (lines === 0) { baseScore = 100; }
            else if (lines === 1) { baseScore = 200; isB2B = true; }
            else if (lines === 2) { baseScore = 400; isB2B = true; }
        } else {
            if (lines === 1) baseScore = 100;
            else if (lines === 2) baseScore = 300;
            else if (lines === 3) baseScore = 500;
            else if (lines === 4) {
                baseScore = 800;
                isB2B = true;
            }
        }

        baseScore *= this.player.level;

        if (isB2B) {
            if (this.player.backToBack) {
                baseScore *= 1.5;
            }
            this.player.backToBack = true;
        } else if (lines > 0) {
            this.player.backToBack = false;
        }

        this.player.score += baseScore;

        // Combo
        if (lines > 0) {
            if (this.player.combo >= 0) {
                this.player.score += 50 * this.player.combo * this.player.level;
            }
            this.player.combo++;
        }

        // Level up
        this.player.lines += lines;
        this.player.totalLines += lines;

        // Calculate Level: Start Level + (Total Lines / 10)
        const newLevel = this.startLevel + Math.floor(this.player.totalLines / 10);

        if (newLevel > this.player.level) {
            this.player.level = newLevel;
            this.updateGravity();
            this.soundManager.playLevelUp();
        }
        this.updateScore();
    }

    updateGravity() {
        // Difficulty Curve
        // Lv 1-10: Speed increases (Standard)
        // Lv 11+: 20G (Instant Drop)

        if (this.player.level <= 10) {
            // Standard Gravity: 50ms decrease per level from 1000ms? No, previous was 50 - (level-1)*5 which is super fast.
            // Let's stick to the previous formula but cap it at Lv 10.
            // Wait, previous formula was: Math.max(16, 50 - (this.player.level - 1) * 5);
            // That starts at 50ms (very fast) and reaches 16ms at Level 8.
            // User said "falling speed at start is already very fast".
            // Let's keep that logic for Lv 1-10.
            this.dropInterval = Math.max(16, 50 - (this.player.level - 1) * 5);
            this.lockDelay = 500;
        } else {
            // Lv 11+: 20G Mode
            this.dropInterval = 0;

            // Lock Delay Curve
            if (this.player.level <= 23) {
                // Lv 11-23: Decrease from 500ms to 400ms
                // Range: 13 levels. Drop: 100ms. ~7.7ms per level.
                const progress = (this.player.level - 10) / 13;
                this.lockDelay = 500 - (100 * progress);
            } else {
                // Lv 24+: Drastic drop to 300ms (or lower)
                // Let's drop to 300ms immediately at Lv 24 and decrease slightly more?
                // User said "max 350ms".
                // Let's set Lv 24 to 350ms and decrease to 250ms by Lv 30.
                const progress = Math.min(1, (this.player.level - 23) / 7);
                this.lockDelay = 350 - (100 * progress);
            }
        }
    }

    victory() {
        this.isRunning = false;
        this.gameOver = true; // Technically game over, but won
        document.getElementById('victory-overlay').classList.remove('hidden');
        document.getElementById('victory-score').innerText = `SCORE: ${Math.floor(this.player.score)}`;
        document.getElementById('victory-time').innerText = `TIME: ${document.getElementById('time-display').innerText}`;
        this.soundManager.playLevelUp(); // Or some victory sound
    }

    updateScore() {
        const scoreElement = document.getElementById('score-display');
        const levelElement = document.getElementById('level-display');
        const linesElement = document.getElementById('lines-display');
        if (scoreElement) scoreElement.innerText = `SCORE: ${Math.floor(this.player.score)}`;
        if (levelElement) levelElement.innerText = `LEVEL: ${this.player.level}`;
        if (linesElement) linesElement.innerText = `LINES: ${this.player.totalLines} / 300`;
    }

    updateTimer() {
        const timeElement = document.getElementById('time-display');
        if (!timeElement) return;

        const totalSeconds = this.elapsedTime / 1000;
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        const centiseconds = Math.floor((totalSeconds % 1) * 100);

        const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${centiseconds.toString().padStart(2, '0')}`;
        timeElement.innerText = formattedTime;
    }

    getGhostPosition() {
        if (!this.player.piece) return null;

        let ghostY = this.player.pos.y;
        while (!this.collide(this.player.pos.x, ghostY + 1, this.player.piece.orientation)) {
            ghostY++;
        }
        return { x: this.player.pos.x, y: ghostY };
    }

    setTheme(isDark) {
        this.renderer.setTheme(isDark);
    }

    restart(startLevel = 1) {
        document.getElementById('game-over-overlay').classList.add('hidden');
        document.getElementById('highscore-form').classList.add('hidden');
        document.getElementById('score-submit-success').classList.add('hidden');
        document.getElementById('victory-overlay').classList.add('hidden');
        this.start(startLevel);
    }

    refillBag() {
        const pieces = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];
        // Fisher-Yates Shuffle
        for (let i = pieces.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [pieces[i], pieces[j]] = [pieces[j], pieces[i]];
        }
        this.bag.push(...pieces);
    }

    getPreviewPieces(count = 5) {
        return this.bag.slice(0, count);
    }

    playerHold() {
        if (this.gameOver || !this.player.canHold || !this.player.piece) return;

        this.player.canHold = false;

        if (this.player.holdPiece === null) {
            this.player.holdPiece = this.player.piece.type;
            this.spawnPiece();
        } else {
            const temp = this.player.piece.type;
            this.player.piece = new Tetromino(this.player.holdPiece);
            this.player.holdPiece = temp;

            this.player.pos = { x: 4, y: 18 };
            this.player.piece.orientation = 0;

            if (this.collide(this.player.pos.x, this.player.pos.y, this.player.piece.orientation)) {
                this.gameOver = true;
                this.isRunning = false;
                document.getElementById('game-over-overlay').classList.remove('hidden');
                document.getElementById('final-score').innerText = `SCORE: ${this.player.score}`;
                return;
            }

            // In 20G mode, immediately drop to the floor
            if (this.dropInterval === 0) {
                while (this.playerDrop()) {}
            }

            const isGrounded = this.collide(this.player.pos.x, this.player.pos.y + 1, this.player.piece.orientation);

            if (isGrounded) {
                this.isLocking = true;
                this.lockCounter = 0;
                this.lockMoveResetCount = 0;
            } else {
                this.isLocking = false;
                this.lockCounter = 0;
                this.lockMoveResetCount = 0;
            }
        }
    }

    isTSpin() {
        if (this.player.piece.type !== 'T' || this.player.lastAction !== 'rotate') {
            return false;
        }

        const x = this.player.pos.x;
        const y = this.player.pos.y;

        const isBlocked = (cx, cy) =>
            cx < 0 || cx >= this.BOARD_WIDTH || cy >= this.BOARD_HEIGHT ||
            (cy >= 0 && this.board[cy][cx] !== 0);

        // Corners: [top-left, top-right, bot-left, bot-right]
        const c = [
            isBlocked(x - 1, y - 1),
            isBlocked(x + 1, y - 1),
            isBlocked(x - 1, y + 1),
            isBlocked(x + 1, y + 1)
        ];

        if (c.filter(Boolean).length < 3) return false;

        // Front corners face the T's nose direction per orientation
        const orient = this.player.piece.orientation;
        let frontA, frontB;
        if      (orient === 0) { frontA = c[0]; frontB = c[1]; } // nose up:    top-left,  top-right
        else if (orient === 1) { frontA = c[1]; frontB = c[3]; } // nose right: top-right, bot-right
        else if (orient === 2) { frontA = c[2]; frontB = c[3]; } // nose down:  bot-left,  bot-right
        else                   { frontA = c[0]; frontB = c[2]; } // nose left:  top-left,  bot-left

        return (frontA && frontB) ? 'full' : 'mini';
    }

    pause() {
        this.isPaused = true;
        this.pauseStartTime = performance.now();
    }

    resume() {
        this.isPaused = false;
        this.totalPausedTime += performance.now() - this.pauseStartTime;
    }
}
