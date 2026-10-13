export default class Renderer {
    constructor(ctx) {
        this.ctx = ctx;
        this.blockSize = 35;
        this.boardOffset = { x: 300, y: 80 };
        this.isDark = false;
        this.showLineClearEffect = true;
        this.ICON_MAP = {
            'Single': 'looks_one',
            'Double': 'looks_two',
            'Triple': 'looks_3',
            'Quad': 'filter_4',
            'Twist_Single': 'sync',
            'Twist_Double': 'autorenew',
            'Twist_Triple': 'rotate_right'
        };
    }

    setTheme(isDark) {
        this.isDark = isDark;
    }

    draw(board, player, game, shakeOffset = { x: 0, y: 0 }, particles = [], bloomIntensity = 0, popups = []) {
        // Strict 2-Color Palette
        // Light: BG = Cream (#fdf6e3), FG = Coal (#121212)
        // Dark: BG = Coal (#121212), FG = Cream (#fdf6e3)

        const CREAM = '#fdf6e3';
        const COAL = '#121212';

        const bgColor = this.isDark ? COAL : CREAM;
        const fgColor = this.isDark ? CREAM : COAL;

        // Grid is same as FG but maybe thinner lines, but user said "same for all... box background".
        // To keep it strictly 2-color, we use FG for grid lines.
        const gridColor = fgColor;
        const pieceColor = fgColor;

        // Shadow/Glow: Use FG color with transparency for glow
        const shadowColor = this.isDark ? 'rgba(253, 246, 227, 0.5)' : 'rgba(18, 18, 18, 0.3)';

        // Clear canvas
        this.ctx.fillStyle = bgColor;
        this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

        this.ctx.save();
        // Apply shake
        this.ctx.translate(shakeOffset.x, shakeOffset.y);

        // Draw Board Background
        this.ctx.fillStyle = bgColor;
        this.ctx.fillRect(
            this.boardOffset.x,
            this.boardOffset.y,
            10 * this.blockSize,
            20 * this.blockSize
        );

        // Draw Grid Lines
        this.ctx.strokeStyle = gridColor;
        this.ctx.lineWidth = 1; // Keep thin for elegance
        this.ctx.shadowBlur = 0;

        // Draw vertical lines
        for (let i = 0; i <= 10; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.boardOffset.x + i * this.blockSize, this.boardOffset.y);
            this.ctx.lineTo(this.boardOffset.x + i * this.blockSize, this.boardOffset.y + 20 * this.blockSize);
            this.ctx.stroke();
        }
        // Draw horizontal lines
        for (let i = 0; i <= 20; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.boardOffset.x, this.boardOffset.y + i * this.blockSize);
            this.ctx.lineTo(this.boardOffset.x + 10 * this.blockSize, this.boardOffset.y + i * this.blockSize);
            this.ctx.stroke();
        }

        // Setup Bloom Effect
        const currentBlur = 15 + bloomIntensity;
        this.ctx.shadowBlur = currentBlur;
        this.ctx.shadowColor = shadowColor;

        // Draw Board Blocks
        const VISIBLE_START = 20;

        board.forEach((row, y) => {
            if (y < VISIBLE_START) return;
            row.forEach((value, x) => {
                if (value !== 0) {
                    this.drawBlock(x, y - VISIBLE_START, pieceColor);
                }
            });
        });

        // Draw Ghost Piece
        if (game && player && player.piece) {
            const ghostPos = game.getGhostPosition();
            if (ghostPos) {
                this.drawGhostPiece(ghostPos, player.piece, pieceColor);
            }
        }

        // Draw Player Piece
        if (player && player.piece) {
            const blocks = player.piece.getBlocks(player.pos.x, player.pos.y, player.piece.orientation);
            blocks.forEach(block => {
                if (block.y >= VISIBLE_START) {
                    this.drawBlock(block.x, block.y - VISIBLE_START, pieceColor);
                }
            });
        }

        // Draw Particles
        this.drawParticles(particles);

        // Draw Clearing Rows Effect (inside shake context so it moves with the board)
        if (this.showLineClearEffect && game && game.clearingRows && game.clearingRows.length > 0) {
            const opacity = game.clearAnimationTimer / game.CLEAR_ANIMATION_DURATION;

            if (this.isDark) {
                this.ctx.fillStyle = `rgba(18, 18, 18, ${opacity * 0.85})`;
                this.ctx.shadowColor = '#121212';
            } else {
                this.ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
                this.ctx.shadowColor = 'white';
            }

            this.ctx.shadowBlur = 20;

            game.clearingRows.forEach(y => {
                if (y >= VISIBLE_START) {
                    this.ctx.fillRect(
                        this.boardOffset.x,
                        this.boardOffset.y + (y - VISIBLE_START) * this.blockSize,
                        10 * this.blockSize,
                        this.blockSize
                    );
                }
            });
        }

        // Reset shadow
        this.ctx.shadowBlur = 0;

        this.ctx.restore(); // Restore context to remove shake for UI elements

        // Draw preview and hold (hidden during countdown)
        if (game && !game.isCountingDown) {
            this.drawPreview(game.getPreviewPieces(5), fgColor);
            this.drawHold(player.holdPiece, fgColor);
        }

        // Draw Stats
        if (game && !game.isCountingDown) {
            this.drawStats(player, game, fgColor, bgColor);
        }

        // Draw Popups
        this.drawPopups(popups, fgColor, bgColor);

        // Draw Countdown
        if (game && game.isCountingDown) {
            const progress = Math.min(1, game.countdownTimer / game.COUNTDOWN_STEP_DURATION);
            this.drawCountdown(game.countdownStep, progress, fgColor, bgColor);
        }
    }

    drawCountdown(step, progress, fgColor, bgColor) {
        // Simple fade: in 0–20%, hold 20–75%, out 75–100%
        let opacity;
        if (progress < 0.2)       opacity = progress / 0.2;
        else if (progress > 0.75) opacity = 1 - (progress - 0.75) / 0.25;
        else                      opacity = 1;
        opacity = Math.max(0, Math.min(1, opacity));

        const text = step === 0 ? 'GO' : String(step);
        const cx = this.boardOffset.x + (10 * this.blockSize) / 2;
        const cy = this.boardOffset.y + (20 * this.blockSize) / 2;

        this.ctx.save();
        this.ctx.globalAlpha = opacity;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.font = `bold ${step === 0 ? 72 : 110}px "Roboto", monospace`;

        // Thin bg outline for readability over the grid
        this.ctx.lineWidth = 6;
        this.ctx.strokeStyle = bgColor;
        this.ctx.strokeText(text, cx, cy);

        this.ctx.fillStyle = fgColor;
        this.ctx.fillText(text, cx, cy);
        this.ctx.restore();
    }

    drawParticles(particles) {
        if (!particles) return;

        this.ctx.save();
        particles.forEach(p => {
            this.ctx.globalAlpha = p.life;
            this.ctx.fillStyle = p.color;

            // Add glow for sparks
            this.ctx.shadowBlur = 5;
            this.ctx.shadowColor = p.color;

            const x = this.boardOffset.x + p.x * this.blockSize;
            const y = this.boardOffset.y + (p.y - 20) * this.blockSize;
            const size = p.size * this.blockSize;

            this.ctx.beginPath();
            this.ctx.arc(x, y, size / 2, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.restore();
    }

    drawBlock(x, y, color) {
        this.ctx.fillStyle = color;
        // Draw solid block without gap (minimalist)
        const gap = 0;
        this.ctx.fillRect(
            this.boardOffset.x + x * this.blockSize + gap,
            this.boardOffset.y + y * this.blockSize + gap,
            this.blockSize - gap * 2,
            this.blockSize - gap * 2
        );
    }

    drawGhostPiece(pos, piece, color) {
        const VISIBLE_START = 20;
        const blocks = piece.getBlocks(pos.x, pos.y, piece.orientation);

        this.ctx.save();
        this.ctx.globalAlpha = 0.2; // Semi-transparent

        blocks.forEach(block => {
            if (block.y >= VISIBLE_START) {
                this.drawBlock(block.x, block.y - VISIBLE_START, color);
            }
        });

        this.ctx.restore();
    }

    drawPreview(previewPieces, color) {
        const startX = 665;
        const startY = 120;

        this.ctx.fillStyle = color;
        this.ctx.font = 'bold 24px "Roboto", monospace';
        this.ctx.fillText('NEXT', startX, startY - 20);

        const offsets = {
            'I': [[0, 1], [1, 1], [2, 1], [3, 1]],
            'O': [[1, 0], [2, 0], [1, 1], [2, 1]],
            'T': [[1, 0], [0, 1], [1, 1], [2, 1]],
            'J': [[0, 0], [0, 1], [1, 1], [2, 1]],
            'L': [[2, 0], [0, 1], [1, 1], [2, 1]],
            'S': [[1, 0], [2, 0], [0, 1], [1, 1]],
            'Z': [[0, 0], [1, 0], [1, 1], [2, 1]]
        };

        const smallBlockSize = 25;

        // Add glow for preview
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = this.isDark ? 'rgba(253, 246, 227, 0.3)' : 'rgba(18, 18, 18, 0.2)';

        previewPieces.forEach((type, index) => {
            const yOffset = startY + index * 80;
            const blocks = offsets[type] || [];

            this.ctx.fillStyle = color;
            blocks.forEach(([bx, by]) => {
                // Draw solid block without gap
                this.ctx.fillRect(
                    startX + bx * smallBlockSize,
                    yOffset + by * smallBlockSize,
                    smallBlockSize,
                    smallBlockSize
                );
            });
        });
        this.ctx.shadowBlur = 0;
    }

    drawHold(holdPiece, color) {
        const startX = 80;
        const startY = 120;

        this.ctx.fillStyle = color;
        this.ctx.font = 'bold 24px "Roboto", monospace';
        this.ctx.fillText('HOLD', startX, startY - 20);

        if (!holdPiece) return;

        const offsets = {
            'I': [[0, 1], [1, 1], [2, 1], [3, 1]],
            'O': [[1, 0], [2, 0], [1, 1], [2, 1]],
            'T': [[1, 0], [0, 1], [1, 1], [2, 1]],
            'J': [[0, 0], [0, 1], [1, 1], [2, 1]],
            'L': [[2, 0], [0, 1], [1, 1], [2, 1]],
            'S': [[1, 0], [2, 0], [0, 1], [1, 1]],
            'Z': [[0, 0], [1, 0], [1, 1], [2, 1]]
        };

        const smallBlockSize = 25;
        const blocks = offsets[holdPiece] || [];

        // Add glow for hold
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = this.isDark ? 'rgba(253, 246, 227, 0.3)' : 'rgba(18, 18, 18, 0.2)';

        this.ctx.fillStyle = color;
        blocks.forEach(([bx, by]) => {
            // Draw solid block without gap
            this.ctx.fillRect(
                startX + bx * smallBlockSize,
                startY + by * smallBlockSize,
                smallBlockSize,
                smallBlockSize
            );
        });
        this.ctx.shadowBlur = 0;
    }

    drawStats(player, game, fgColor, bgColor) {
        const x = 50;
        let y = 260;
        const lineH = 66;

        // Format time
        const totalSeconds = (game.elapsedTime || 0) / 1000;
        const mins = Math.floor(totalSeconds / 60);
        const secs = Math.floor(totalSeconds % 60);
        const cs = Math.floor((totalSeconds % 1) * 100);
        const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;

        const stats = [
            { label: 'SCORE', value: Math.floor(player.score).toLocaleString() },
            { label: 'LEVEL', value: String(player.level) },
            { label: 'LINES', value: `${player.totalLines} / 300` },
            { label: 'TIME',  value: timeStr },
        ];

        this.ctx.save();
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'top';
        this.ctx.shadowBlur = 0;

        stats.forEach(({ label, value }) => {
            // Label
            this.ctx.font = 'bold 15px "Roboto", monospace';
            this.ctx.fillStyle = fgColor;
            this.ctx.globalAlpha = 0.5;
            this.ctx.fillText(label, x, y);

            // Value
            this.ctx.font = 'bold 30px "Roboto", monospace';
            this.ctx.globalAlpha = 1;
            this.ctx.fillText(value, x, y + 18);

            y += lineH;
        });

        this.ctx.restore();
    }

    drawPopups(popups, textColor, strokeColor) {
        if (!popups) return;

        this.ctx.save();
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        popups.forEach(p => {
            this.ctx.globalAlpha = p.life;

            // Check if text maps to an icon
            const iconName = this.ICON_MAP[p.text];

            if (iconName) {
                // Draw Icon
                this.ctx.font = '60px "Material Icons"';

                // Stroke (Background color for outline effect)
                this.ctx.lineWidth = 3;
                this.ctx.strokeStyle = strokeColor;
                this.ctx.strokeText(iconName, p.x, p.y);

                // Fill (Foreground color)
                this.ctx.fillStyle = textColor;
                this.ctx.fillText(iconName, p.x, p.y);
            } else {
                // Draw Text
                this.ctx.font = 'bold 40px "Roboto", monospace';

                // Stroke
                this.ctx.lineWidth = 3;
                this.ctx.strokeStyle = strokeColor;
                this.ctx.strokeText(p.text, p.x, p.y);

                // Fill
                this.ctx.fillStyle = textColor;
                this.ctx.fillText(p.text, p.x, p.y);
            }
        });

        this.ctx.restore();
    }
}
