export const DEFAULT_BINDINGS = {
    moveLeft:  'ArrowLeft',
    moveRight: 'ArrowRight',
    softDrop:  'ArrowDown',
    rotateCCW: 'KeyZ',
    rotateCW:  'KeyX',
    hardDrop:  'Space',
    hold:      'KeyC',
};

export default class Input {
    constructor(game) {
        this.game = game;

        // DAS (Delayed Auto Shift) and ARR (Auto Repeat Rate) settings
        this.keys = {
            left: false,
            right: false,
            down: false,
            rotateCCW: false,
            rotateCW: false,
            hold: false,
        };

        this.dasDelay = 100; // Delay before auto-repeat starts (ms)
        this.arrRate = 30;   // Auto-repeat rate (ms between moves)

        this.dasTimer = { left: 0, right: 0, down: 0 };
        this.arrTimer = { left: 0, right: 0, down: 0 };

        this.bindings = { ...DEFAULT_BINDINGS };

        document.addEventListener('keydown', event => {
            if (event.code === 'Escape') {
                if (this.onCaptureKey) {
                    this.onCaptureKey('Escape');
                } else {
                    this.onEscape?.();
                }
                return;
            }

            if (this.onCaptureKey) {
                this.onCaptureKey(event.code);
                return;
            }

            // Track rotate/hold key state for IRS/IHS regardless of game phase
            const b = this.bindings;
            const code = event.code;
            if (code === b.rotateCCW) this.keys.rotateCCW = true;
            if (code === b.rotateCW)  this.keys.rotateCW  = true;
            if (code === b.hold)      this.keys.hold      = true;

            if (!this.game.isRunning || this.game.isCountingDown) return;

            if (code === b.moveLeft) {
                if (!this.keys.left) {
                    this.keys.left = true;
                    this.keys.right = false;
                    this.game.playerMove(-1);
                    this.dasTimer.left = 0;
                    this.arrTimer.left = 0;
                }
            } else if (code === b.moveRight) {
                if (!this.keys.right) {
                    this.keys.right = true;
                    this.keys.left = false;
                    this.game.playerMove(1);
                    this.dasTimer.right = 0;
                    this.arrTimer.right = 0;
                }
            } else if (code === b.softDrop) {
                if (!this.keys.down) {
                    this.keys.down = true;
                    this.game.playerDrop(true);
                    this.dasTimer.down = 0;
                    this.arrTimer.down = 0;
                }
            } else if (code === b.rotateCCW) {
                this.game.playerRotate(-1);
            } else if (code === b.rotateCW) {
                this.game.playerRotate(1);
            } else if (code === b.hardDrop) {
                this.game.playerHardDrop();
            } else if (code === b.hold) {
                this.game.playerHold();
            }
        });

        document.addEventListener('keyup', event => {
            const b = this.bindings;
            const code = event.code;
            if (code === b.moveLeft)   this.keys.left      = false;
            if (code === b.moveRight)  this.keys.right     = false;
            if (code === b.softDrop)   this.keys.down      = false;
            if (code === b.rotateCCW)  this.keys.rotateCCW = false;
            if (code === b.rotateCW)   this.keys.rotateCW  = false;
            if (code === b.hold)       this.keys.hold      = false;
        });
    }

    update(deltaTime) {
        if (!this.game.isRunning) return;

        // Handle left movement with DAS/ARR
        if (this.keys.left) {
            this.dasTimer.left += deltaTime;
            if (this.dasTimer.left >= this.dasDelay) {
                this.arrTimer.left += deltaTime;
                if (this.arrTimer.left >= this.arrRate) {
                    this.game.playerMove(-1);
                    this.arrTimer.left = 0;
                }
            }
        }

        // Handle right movement with DAS/ARR
        if (this.keys.right) {
            this.dasTimer.right += deltaTime;
            if (this.dasTimer.right >= this.dasDelay) {
                this.arrTimer.right += deltaTime;
                if (this.arrTimer.right >= this.arrRate) {
                    this.game.playerMove(1);
                    this.arrTimer.right = 0;
                }
            }
        }

        // Handle down movement with DAS/ARR
        if (this.keys.down) {
            this.dasTimer.down += deltaTime;
            if (this.dasTimer.down >= this.dasDelay) {
                this.arrTimer.down += deltaTime;
                if (this.arrTimer.down >= this.arrRate) {
                    this.game.playerDrop(true);
                    this.arrTimer.down = 0;
                }
            }
        }
    }
}
