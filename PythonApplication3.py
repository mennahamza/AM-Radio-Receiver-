import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
import sounddevice as sd
from scipy.signal import butter, filtfilt, wiener
import threading
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
#  PREMIUM SETTINGS & COLORS
# ══════════════════════════════════════════════════════
SAMPLE_RATE = 44100
DURATION = 2.0

COLORS = {
    'bg': '#020408', 'panel': '#0b1221', 'border': '#1a2a45',
    'text': '#e1e8f5', 'cyan': '#00f2ff', 'gold': '#ffcc00',
    's1': '#ff0055', 's2': '#ff6b00', 's3': '#ffcc00', 
    's4': '#00ffaa', 's5': '#00aaff', 's6': '#9d00ff', 'err': '#ff3333'
}

state = {'fa_hz': 440, 'm': 0.8, 'noise_db': -20}

# ══════════════════════════════════════════════════════
#  SIGNAL ENGINE
# ══════════════════════════════════════════════════════
def generate_full_chain():
    fa, m, ndb = state['fa_hz'], state['m'], state['noise_db']
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    source = np.sin(2 * np.pi * fa * t)
    carrier = np.cos(2 * np.pi * 5000 * t)
    antenna = carrier + np.random.normal(0, 0.4, len(t))
    am_noisy = (1 + m * source) * carrier + np.random.normal(0, 10**(ndb/20) * 2, len(t))
    rectified = np.maximum(am_noisy, 0)
    
    b, a = butter(6, 0.12, btype='low')
    filt = filtfilt(b, a, rectified)
    recovered = wiener(filt)
    recovered = (recovered - np.mean(recovered))
    recovered /= (np.max(np.abs(recovered)) + 1e-9)
    
    return t, {
        'src': source, 'ant': antenna, 'am': am_noisy, 
        'dio': rectified, 'flt': filt, 'rec': recovered
    }

# ══════════════════════════════════════════════════════
#  DASHBOARD LAYOUT (Grid with Comparison Row)
# ══════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 11), facecolor=COLORS['bg'])
# تنظيم الشبكة: صفين للخطوات، وصف ثالث كبير للمقارنة
gs = GridSpec(4, 3, figure=fig, hspace=0.7, wspace=0.3, bottom=0.2, top=0.95)

def apply_ax_style(ax, title, color):
    ax.set_facecolor(COLORS['panel'])
    ax.set_title(title, color=color, fontsize=9, fontweight='bold')
    ax.tick_params(colors='#444', labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor(COLORS['border'])

# المربعات الستة الأولى
axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(6)]
ax_titles = ["01. ANTENNA (STATIC)", "02. AM SIGNAL", "03. DIODE OUT", 
             "04. FILTERED", "05. AMPLIFIED", "06. RECOVERED"]
ax_colors = [COLORS['s1'], COLORS['s2'], COLORS['s3'], COLORS['s4'], COLORS['s5'], COLORS['s6']]

for ax, title, col in zip(axes, ax_titles, ax_colors):
    apply_ax_style(ax, title, col)

# كيرف المقارنة الكبير (يأخذ الصف الثالث والرابع)
ax_cmp = fig.add_subplot(gs[2:4, :])
apply_ax_style(ax_cmp, "ULTIMATE COMPARISON: ORIGINAL SOURCE vs. RECOVERED OUTPUT", COLORS['cyan'])

# ── Initial Data ──
t, d = generate_full_chain()
W = 1200; tw = t[:W]*1000
lines = []
keys = ['ant', 'am', 'dio', 'flt', 'rec', 'rec']

for i in range(6):
    ln, = axes[i].plot(tw, d[keys[i]][:W], color=ax_colors[i], lw=1)
    lines.append(ln)

# رسم المقارنة في الكيرف الكبير
ln_orig, = ax_cmp.plot(tw, d['src'][:W], color=COLORS['s4'], lw=2, label='Original Source', alpha=0.6)
ln_reco, = ax_cmp.plot(tw, d['rec'][:W], color=COLORS['cyan'], lw=1.5, ls='--', label='Recovered Signal')
ln_diff, = ax_cmp.plot(tw, (d['src'][:W] - d['rec'][:W]), color=COLORS['err'], lw=0.8, label='Extraction Error', alpha=0.5)
ax_cmp.legend(loc='upper right', fontsize=8, facecolor=COLORS['bg'], labelcolor='white')

# ══════════════════════════════════════════════════════
#  AUDIO BUTTONS (Localized)
# ══════════════════════════════════════════════════════
def play_stage(key, vol=0.3):
    _, d_play = generate_full_chain()
    sig = (d_play[key] / (np.max(np.abs(d_play[key])) + 1e-9)) * vol
    sd.play(sig[:int(SAMPLE_RATE*1.5)], SAMPLE_RATE)

btns = []
for i in range(6):
    pos = axes[i].get_position()
    bx = fig.add_axes([pos.x0 + pos.width*0.2, pos.y0 - 0.05, pos.width*0.6, 0.03])
    b = Button(bx, f'🔊 LISTEN {i+1}', color=COLORS['panel'], hovercolor='#1a2a45')
    b.label.set_color(ax_colors[i])
    b.label.set_fontsize(7)
    b.on_clicked(lambda x, k=keys[i]: threading.Thread(target=play_stage, args=(k,)).start())
    btns.append(b)

# ══════════════════════════════════════════════════════
#  CONTROLS
# ══════════════════════════════════════════════════════
ax_fa = fig.add_axes([0.2, 0.08, 0.25, 0.015], facecolor='#0f1f3d')
sl_fa = Slider(ax=ax_fa, label='Tone (Hz) ', valmin=200, valmax=1200, valinit=440, color=COLORS['cyan'])

ax_no = fig.add_axes([0.55, 0.08, 0.25, 0.015], facecolor='#0f1f3d')
sl_no = Slider(ax=ax_no, label='Noise Level ', valmin=-50, valmax=-5, valinit=-20, color=COLORS['s1'])

def update(v):
    state['fa_hz'], state['noise_db'] = sl_fa.val, sl_no.val
    state['fa_hz'], state['noise_db'] = sl_fa.val, sl_no.val
    _, d2 = generate_full_chain()
    for i, k in enumerate(keys):
        lines[i].set_ydata(d2[k][:W])
    
    # تحديث كيرف المقارنة
    ln_orig.set_ydata(d2['src'][:W])
    ln_reco.set_ydata(d2['rec'][:W])
    ln_diff.set_ydata(d2['src'][:W] - d2['rec'][:W])
    
    fig.canvas.draw_idle()

sl_fa.on_changed(update); sl_no.on_changed(update)

plt.show()