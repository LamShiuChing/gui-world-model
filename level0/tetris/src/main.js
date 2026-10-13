import Game from './engine/Game.js';
import { DEFAULT_BINDINGS } from './engine/Input.js';
const Grid = class { render() {} }, html = s => s; // research copy: leaderboard table stubbed (no bundler)


const canvas = document.getElementById('gameCanvas');
const game = new Game(canvas);
window.__game = game; // research hook: lets the harness read the true game state

// UI Elements
// UI Elements
const menuOverlay = document.getElementById('menu-overlay');
const pauseOverlay = document.getElementById('pause-overlay');
const startBtn = document.getElementById('start-btn');
const restartBtn = document.getElementById('restart-btn');
const backToMenuBtn = document.getElementById('back-to-menu-btn');
const victoryRestartBtn = document.getElementById('victory-restart-btn');
const themeToggle = document.getElementById('theme-toggle');
const themeLabel = document.getElementById('theme-label');
const startLevelInput = document.getElementById('start-level');
const startLevelValue = document.getElementById('start-level-value');

// Start Level Slider
startLevelInput.addEventListener('input', (e) => {
    startLevelValue.innerText = e.target.value;
});

// Theme Handling
themeToggle.addEventListener('change', (e) => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    const isDark = e.target.checked;
    document.body.classList.toggle('dark-theme', isDark);
    themeLabel.innerText = isDark ? 'NIGHT' : 'INK';
    game.setTheme(isDark);
});

// Countdown then start (rendered on canvas)
function startWithCountdown(startLevel) {
    game.startCountdown(startLevel);
}

// Start Game
startBtn.addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    menuOverlay.classList.add('hidden');
    const startLevel = parseInt(startLevelInput.value, 10);
    startWithCountdown(startLevel);
});

// Restart Game (Game Over)
restartBtn.addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    document.getElementById('game-over-overlay').classList.add('hidden');
    document.getElementById('highscore-form').classList.add('hidden');
    document.getElementById('score-submit-success').classList.add('hidden');
    const startLevel = parseInt(startLevelInput.value, 10);
    startWithCountdown(startLevel);
});

// Restart Game (Victory)
victoryRestartBtn.addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    document.getElementById('victory-overlay').classList.add('hidden');
    const startLevel = parseInt(startLevelInput.value, 10);
    startWithCountdown(startLevel);
});

// Back to Menu
backToMenuBtn.addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    document.getElementById('game-over-overlay').classList.add('hidden');
    menuOverlay.classList.remove('hidden');
});

// View Leaderboard from Game Over
document.getElementById('view-leaderboard-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    navigateToSection('leaderboard-section');
});

// ESC Pause
game.input.onEscape = () => {
    if (!settingsOverlay.classList.contains('hidden')) {
        settingsOverlay.classList.add('hidden');
        return;
    }
    if (game.gameOver || !game.isRunning || game.isCountingDown) return;
    if (game.isPaused) {
        game.resume();
        pauseOverlay.classList.add('hidden');
    } else {
        game.pause();
        pauseOverlay.classList.remove('hidden');
    }
};

document.getElementById('resume-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    game.resume();
    pauseOverlay.classList.add('hidden');
});

document.getElementById('pause-restart-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    pauseOverlay.classList.add('hidden');
    game.isRunning = false;
    game.isPaused = false;
    const startLevel = parseInt(startLevelInput.value, 10);
    startWithCountdown(startLevel);
});

document.getElementById('pause-menu-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    pauseOverlay.classList.add('hidden');
    game.isRunning = false;
    game.isPaused = false;
    navigateToSection('play-section');
});

// Settings
const settingsOverlay = document.getElementById('settings-overlay');
const dasSlider = document.getElementById('das-slider');
const arrSlider = document.getElementById('arr-slider');
const shakeSlider = document.getElementById('shake-slider');
const sfxSlider = document.getElementById('sfx-slider');
const lineclearToggle = document.getElementById('lineclear-toggle');

function applySettings(das, arr, shake, sfx, lineclear) {
    game.input.dasDelay = das;
    game.input.arrRate = arr;
    game.shakeMultiplier = shake / 100;
    game.soundManager.setVolume(sfx / 100);
    game.renderer.showLineClearEffect = lineclear;
}

function loadSettings() {
    const das = parseInt(localStorage.getItem('setting_das') ?? '100', 10);
    const arr = parseInt(localStorage.getItem('setting_arr') ?? '30', 10);
    const shake = parseInt(localStorage.getItem('setting_shake') ?? '100', 10);
    const sfx = parseInt(localStorage.getItem('setting_sfx') ?? '100', 10);
    const lineclear = localStorage.getItem('setting_lineclear') !== 'false';

    dasSlider.value = das;
    arrSlider.value = arr;
    shakeSlider.value = shake;
    sfxSlider.value = sfx;
    lineclearToggle.checked = lineclear;

    document.getElementById('das-value-display').innerText = das + ' ms';
    document.getElementById('arr-value-display').innerText = arr === 0 ? '0 ms (instant)' : arr + ' ms';
    document.getElementById('shake-value-display').innerText = shake + ' %';
    document.getElementById('sfx-value-display').innerText = sfx + ' %';

    applySettings(das, arr, shake, sfx, lineclear);
}

function saveAndApply() {
    const das = parseInt(dasSlider.value, 10);
    const arr = parseInt(arrSlider.value, 10);
    const shake = parseInt(shakeSlider.value, 10);
    const sfx = parseInt(sfxSlider.value, 10);
    const lineclear = lineclearToggle.checked;

    localStorage.setItem('setting_das', das);
    localStorage.setItem('setting_arr', arr);
    localStorage.setItem('setting_shake', shake);
    localStorage.setItem('setting_sfx', sfx);
    localStorage.setItem('setting_lineclear', lineclear);

    document.getElementById('das-value-display').innerText = das + ' ms';
    document.getElementById('arr-value-display').innerText = arr === 0 ? '0 ms (instant)' : arr + ' ms';
    document.getElementById('shake-value-display').innerText = shake + ' %';
    document.getElementById('sfx-value-display').innerText = sfx + ' %';

    applySettings(das, arr, shake, sfx, lineclear);
}

dasSlider.addEventListener('input', saveAndApply);
arrSlider.addEventListener('input', saveAndApply);
shakeSlider.addEventListener('input', saveAndApply);
sfxSlider.addEventListener('input', saveAndApply);
lineclearToggle.addEventListener('change', saveAndApply);

function openSettings() {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    settingsOverlay.classList.remove('hidden');
}

document.getElementById('menu-settings-btn').addEventListener('click', openSettings);
document.getElementById('pause-settings-btn').addEventListener('click', openSettings);

document.getElementById('settings-close-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    // Cancel any in-progress keybind capture
    if (capturingBtn) {
        capturingBtn.textContent = codeToDisplay(game.input.bindings[capturingBtn.dataset.action]);
        capturingBtn.classList.remove('capturing');
        game.input.onCaptureKey = null;
        capturingBtn = null;
    }
    settingsOverlay.classList.add('hidden');
});

loadSettings();

document.getElementById('reset-general-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    localStorage.removeItem('setting_das');
    localStorage.removeItem('setting_arr');
    localStorage.removeItem('setting_shake');
    localStorage.removeItem('setting_sfx');
    localStorage.removeItem('setting_lineclear');
    loadSettings();
});

// Settings tabs
document.querySelectorAll('.settings-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const target = tab.dataset.tab;
        document.getElementById('tab-general').classList.toggle('hidden', target !== 'general');
        document.getElementById('tab-keybinds').classList.toggle('hidden', target !== 'keybinds');
    });
});

// Keybinds
const ACTION_LABELS = {
    moveLeft:  'MOVE LEFT',
    moveRight: 'MOVE RIGHT',
    softDrop:  'SOFT DROP',
    rotateCCW: 'ROTATE CCW',
    rotateCW:  'ROTATE CW',
    hardDrop:  'HARD DROP',
    hold:      'HOLD',
};

function codeToDisplay(code) {
    if (code === 'Space') return 'SPACE';
    if (code === 'ArrowLeft')  return '←';
    if (code === 'ArrowRight') return '→';
    if (code === 'ArrowUp')    return '↑';
    if (code === 'ArrowDown')  return '↓';
    if (code.startsWith('Key'))   return code.slice(3);
    if (code.startsWith('Digit')) return code.slice(5);
    return code;
}

function loadKeybinds() {
    const saved = localStorage.getItem('setting_keybinds');
    const bindings = saved ? { ...DEFAULT_BINDINGS, ...JSON.parse(saved) } : { ...DEFAULT_BINDINGS };
    game.input.bindings = bindings;
    updateKeybindButtons(bindings);
}

function saveKeybinds(bindings) {
    localStorage.setItem('setting_keybinds', JSON.stringify(bindings));
    game.input.bindings = { ...bindings };
    updateKeybindButtons(bindings);
}

function updateKeybindButtons(bindings) {
    document.querySelectorAll('.key-btn').forEach(btn => {
        const action = btn.dataset.action;
        btn.textContent = codeToDisplay(bindings[action] || DEFAULT_BINDINGS[action]);
        btn.classList.remove('capturing');
    });
}

let capturingBtn = null;

document.querySelectorAll('.key-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (capturingBtn) {
            capturingBtn.textContent = codeToDisplay(game.input.bindings[capturingBtn.dataset.action]);
            capturingBtn.classList.remove('capturing');
            game.input.onCaptureKey = null;
        }
        capturingBtn = btn;
        btn.textContent = '...';
        btn.classList.add('capturing');

        game.input.onCaptureKey = (code) => {
            game.input.onCaptureKey = null;
            capturingBtn = null;
            btn.classList.remove('capturing');

            if (code === 'Escape') {
                btn.textContent = codeToDisplay(game.input.bindings[btn.dataset.action]);
                return;
            }

            const action = btn.dataset.action;
            const bindings = { ...game.input.bindings };

            // Check conflict
            const conflict = Object.entries(bindings).find(([a, c]) => c === code && a !== action);
            if (conflict) {
                showToast(`Already bound to ${ACTION_LABELS[conflict[0]]}`);
                btn.textContent = codeToDisplay(bindings[action]);
                return;
            }

            bindings[action] = code;
            saveKeybinds(bindings);
        };
    });
});

document.getElementById('reset-keybinds-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    if (capturingBtn) {
        capturingBtn.classList.remove('capturing');
        game.input.onCaptureKey = null;
        capturingBtn = null;
    }
    saveKeybinds({ ...DEFAULT_BINDINGS });
    localStorage.removeItem('setting_keybinds');
});

loadKeybinds();

// Toast Notification
function showToast(message) {
    const toast = document.getElementById('toast-notification');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove('hidden');
    toast.classList.add('visible');
    clearTimeout(toast._hideTimeout);
    toast._hideTimeout = setTimeout(() => {
        toast.classList.remove('visible');
        toast.classList.add('hidden');
    }, 3000);
}

// Navigation Logic
const navItems = document.querySelectorAll('.nav-item');
const pageSections = document.querySelectorAll('.page-section');
// Donate Confirmation Handler
const confirmDonateBtn = document.getElementById('confirm-donate-btn');
if (confirmDonateBtn) {
    confirmDonateBtn.addEventListener('click', (e) => {
        e.preventDefault();
        try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
        // Open Stripe Payment Link in new tab
        window.open('https://buy.stripe.com/14AcN4aiw4EJ5kO0ydcZa00', '_blank');
    });
}

// Check for Donation Status
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('status') === 'success') {
    showToast('Thank you for your donation!');
    // Clean up URL
    window.history.replaceState({}, document.title, "/");
} else if (urlParams.get('status') === 'canceled') {
    // alert('Donation canceled.');
    window.history.replaceState({}, document.title, "/");
}

const navigateToSection = (targetId) => {
    // Remove active class from all nav items
    navItems.forEach(nav => {
        if (nav.id !== 'donate-btn') nav.classList.remove('active');
    });

    // Highlight matching nav item if present
    const matchingNav = Array.from(navItems).find(nav => nav.getAttribute('data-target') === targetId);
    if (matchingNav) {
        matchingNav.classList.add('active');
    }

    // Hide all sections
    pageSections.forEach(section => section.classList.add('hidden'));
    pageSections.forEach(section => section.classList.remove('active'));

    // Show target section
    const targetSection = document.getElementById(targetId);
    if (targetSection) {
        targetSection.classList.remove('hidden');
        targetSection.classList.add('active');
        if (targetId === 'leaderboard-section') {
            fetchLeaderboard();
        }
    }

    // Overlay Handling
    if (targetId !== 'play-section') {
        // Hide all overlays when not in play section
        menuOverlay.classList.add('hidden');
        document.getElementById('game-over-overlay').classList.add('hidden');
        document.getElementById('victory-overlay').classList.add('hidden');
        pauseOverlay.classList.add('hidden');

        // Pause game if running
        if (game.isRunning && !game.isPaused && !game.gameOver) {
            game.pause();
        }
    } else {
        // In Play Section
        if (game.gameOver) {
            // Show game over overlay if game is over
            document.getElementById('game-over-overlay').classList.remove('hidden');
        } else if (!game.isRunning) {
            // Show start menu if game hasn't started and not game over
            menuOverlay.classList.remove('hidden');
        } else if (game.isRunning && game.isPaused) {
            // Resume game if paused
            game.resume();
        }
    }
};

navItems.forEach(item => {
    // if (item.id === 'donate-btn') return; // Skip donate button for nav logic
    item.addEventListener('click', (e) => {
        e.preventDefault();
        try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
        const targetId = item.getAttribute('data-target');
        navigateToSection(targetId);
    });
});

// Nav Logo Click Handler
const navLogo = document.querySelector('.nav-logo');
if (navLogo) {
    navLogo.addEventListener('click', (e) => {
        e.preventDefault();
        try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
        navigateToSection('play-section');
    });
}

// Leaderboard Logic
const leaderboardList = document.getElementById('leaderboard-list');
const leaderboardLoading = document.getElementById('leaderboard-loading');
const leaderboardContent = document.getElementById('leaderboard-content');

document.getElementById('leaderboard-refresh-btn').addEventListener('click', () => {
    try { game.soundManager.playClick(); } catch (err) { console.warn(err); }
    fetchLeaderboard();
});

let leaderboardGrid = null;

async function fetchLeaderboard() {
    if (!leaderboardList) return;

    leaderboardLoading.classList.remove('hidden');
    leaderboardContent.classList.add('hidden');

    // Clear previous content if any (though Grid.js handles its own container)
    leaderboardList.innerHTML = '';

    try {
        let data = [];
        try {
            const response = await fetch('/api/leaderboard');
            if (!response.ok) throw new Error('Network response was not ok');
            data = await response.json();
        } catch (err) {
            console.warn('Using mock data:', err);
            data = [
                { username: 'TETRIS_GOD', score: 999999, duration: 1200, country: 'JP' },
                { username: 'SPEED_DEMON', score: 850000, duration: 900, country: 'US' },
                { username: 'BLOCK_MASTER', score: 720000, duration: 1100, country: 'KR' },
                { username: 'CASUAL_PLAYER', score: 150000, duration: 300, country: 'BR' },
                { username: 'NEWBIE', score: 50000, duration: 120, country: 'CA' },
            ];
        }

        // Transform data for Grid.js
        const gridData = data.map((entry, index) => {
            const minutes = Math.floor((entry.duration || 0) / 60);
            const seconds = (entry.duration || 0) % 60;
            const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;

            return [
                index + 1,
                entry.username,
                entry.score.toLocaleString(),
                timeStr,
                entry.country || '-'
            ];
        });

        if (leaderboardGrid) {
            leaderboardGrid.updateConfig({
                data: gridData
            }).forceRender();
        } else {
            leaderboardGrid = new Grid({
                columns: ['Rank', 'Player', 'Score', 'Time', 'Country'],
                data: gridData,
                search: true,
                sort: true,
                pagination: {
                    limit: 10
                },
                className: {
                    table: 'custom-grid-table',
                    th: 'custom-grid-th',
                    td: 'custom-grid-td'
                }
            }).render(leaderboardList);
        }

        leaderboardLoading.classList.add('hidden');
        leaderboardContent.classList.remove('hidden');

    } catch (error) {
        console.error('Critical error loading leaderboard:', error);
        leaderboardLoading.innerText = 'Failed to load leaderboard.';
    }
}

// Submit Score
const submitScoreBtn = document.getElementById('submit-score-btn');
const playerNameInput = document.getElementById('player-name');
const highscoreForm = document.getElementById('highscore-form');

if (submitScoreBtn) {
    submitScoreBtn.addEventListener('click', async () => {
        const username = playerNameInput.value.trim();
        if (!username) return;

        submitScoreBtn.disabled = true;
        submitScoreBtn.innerText = 'SUBMITTING...';

        try {
            const country = (navigator.language || '').split('-')[1] || '';
            await fetch('/api/leaderboard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username,
                    score: Math.floor(game.player.score),
                    duration: Math.floor(game.elapsedTime / 1000),
                    country
                })
            });

            highscoreForm.classList.add('hidden');
            document.getElementById('score-submit-success').classList.remove('hidden');
        } catch (error) {
            console.error('Failed to submit score:', error);
            showToast('Failed to submit score.');
        } finally {
            submitScoreBtn.disabled = false;
            submitScoreBtn.innerText = 'SUBMIT SCORE';
        }
    });
}
