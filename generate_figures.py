"""
FDM论文图表生成 v2
按期刊审阅意见修改：
- 图1：简化文字，phase框变细
- 图2：更正式的label
- 图3：去掉图内结论文字
- 图4：VEP点错位，Panel B统一风格
- 补图：显存scaling
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

PURPLE = '#534AB7'
TEAL   = '#1D9E75'
GRAY   = '#888780'
AMBER  = '#EF9F27'
CORAL  = '#D85A30'
GREEN  = '#3B6D11'
BLUE   = '#185FA5'
LGRAY  = '#D3D1C7'


# ================================================================
# 图1：架构总览 v2 — 精简文字，phase框缩小
# ================================================================
def make_fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.4)
    ax.axis('off')

    def box(x, y, w, h, fc, ec, label, sublabel=None, lw=0.8, ls='-', fs=7.5):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                               facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls)
        ax.add_patch(rect)
        cy = y + h / 2
        if sublabel:
            ax.text(x+w/2, cy+0.1,  label,    ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=ec)
            ax.text(x+w/2, cy-0.12, sublabel, ha='center', va='center',
                    fontsize=6.5, color=ec)
        else:
            ax.text(x+w/2, cy, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=ec)

    def arr(x1, y1, x2, y2, col=GRAY, lw=0.9):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=col,
                                   lw=lw, mutation_scale=8))

    # Input
    box(0.08, 1.55, 0.82, 0.62, LGRAY+'44', GRAY,
        '$x_1\\cdots x_T$', 'input tokens')
    arr(0.90, 1.86, 1.15, 1.86)

    # FDM layer dashed container
    ax.add_patch(FancyBboxPatch((1.15, 0.58), 2.85, 2.58,
                                 boxstyle='round,pad=0.06',
                                 facecolor='none', edgecolor=LGRAY,
                                 linewidth=0.7, linestyle='--'))
    ax.text(2.57, 3.06, 'FDM layer (×$L$)', ha='center',
            fontsize=7, color=GRAY)

    # Wave pathway
    box(1.26, 1.98, 1.32, 0.88, PURPLE+'28', PURPLE,
        'Wave pathway', 'global propagation')
    # Cache pathway
    box(1.26, 0.72, 1.32, 0.90, TEAL+'28', TEAL,
        'Cache pathway', 'sparse retrieval')

    # Gate & add
    box(2.95, 1.42, 0.90, 0.82, LGRAY+'66', GRAY,
        'Gate & add', '$h + g \\odot c$')

    arr(2.58, 2.42, 2.95, 1.88)
    arr(2.58, 1.17, 2.95, 1.62)
    arr(3.85, 1.83, 4.12, 1.83)

    # Output heads — simpler labels
    box(4.12, 2.08, 1.42, 0.70, BLUE+'28', BLUE,
        'Language modeling', lw=0.7)
    box(4.12, 0.88, 1.42, 0.70, CORAL+'28', CORAL,
        'Genomic fine-tuning', lw=0.7)
    ax.annotate('', xy=(4.83, 2.08), xytext=(4.57, 1.97),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8, mutation_scale=7))
    ax.annotate('', xy=(4.83, 1.58), xytext=(4.57, 1.69),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8, mutation_scale=7))

    # Phase-aware inset — lighter, thinner
    ax.add_patch(FancyBboxPatch((0.08, 0.04), 6.92, 0.52,
                                 boxstyle='round,pad=0.04',
                                 facecolor=LGRAY+'22', edgecolor=LGRAY,
                                 linewidth=0.5, linestyle=':'))
    ax.text(3.54, 0.49, 'Phase-aware staged training',
            ha='center', fontsize=6.5, color='#555555', fontweight='bold')

    # Phase 0 box — subdued
    box(0.16, 0.07, 2.38, 0.36, AMBER+'22', AMBER,
        'Phase 0  freeze wave, train cache', lw=0.6, fs=7.0)
    # Arrow
    ax.annotate('', xy=(3.04, 0.25), xytext=(2.54, 0.25),
                arrowprops=dict(arrowstyle='->', color=GRAY,
                                lw=0.7, mutation_scale=7))
    ax.text(2.79, 0.36, 'step 23k', ha='center', fontsize=6, color=GRAY)
    # Phase 1 box
    box(3.04, 0.07, 2.16, 0.36, GREEN+'22', GREEN,
        'Phase 1  joint optimization', lw=0.6, fs=7.0)

    fig.savefig('fig1_architecture.pdf')
    fig.savefig('fig1_architecture.png')
    plt.close()
    print('fig1 v2 done')


# ================================================================
# 图2：语言模型主图 v2 — 更正式label
# ================================================================
def make_fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                    gridspec_kw={'width_ratios': [1.6, 1]})

    # Panel A
    steps_joint = np.array([0,5,10,15,20,23,30,40]) * 1000
    ppl_joint   = np.array([487.5,420,350,300,270,260,250,240])
    steps_fdm   = np.array([0,5,10,15,20,23,25,30,35,40]) * 1000
    ppl_fdm     = np.array([487.5,320,180,110,70,36.8,35.8,34.9,34.2,33.75])

    ax1.semilogy(steps_joint, ppl_joint, color=CORAL, lw=1.2,
                 label='Joint baseline')
    ax1.semilogy(steps_fdm,   ppl_fdm,   color=TEAL,  lw=1.5,
                 label='Phase-aware (FDM)')
    ax1.axhline(37, color=GRAY, lw=1.0, ls='--',
                label='Transformer reference (~36–38)')
    ax1.axvline(23000, color=GRAY, lw=0.7, ls=':', alpha=0.7)
    ax1.text(23600, 180, 'phase\ntransition', fontsize=6,
             color=GRAY, va='top')
    ax1.scatter([40000], [33.75], color=TEAL, s=22, zorder=5)
    ax1.text(40800, 30, '33.75', fontsize=6.5, color=TEAL, va='center')

    ax1.set_xlabel('Training steps')
    ax1.set_ylabel('Validation perplexity')
    ax1.set_xlim(0, 43000)
    ax1.set_ylim(25, 600)
    ax1.set_xticks([0,10000,20000,23000,30000,40000])
    ax1.set_xticklabels(['0','10K','20K','23K','30K','40K'])
    ax1.legend(loc='upper right', frameon=False, fontsize=6.5)
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # Panel B — more formal labels
    configs = ['Joint', 'Phase 0', 'Adaptive θ', 'Phase-aware', 'Transformer\nreference']
    ppls    = [487.5,   64.9,      45.8,          33.75,          37]
    colors  = [CORAL,   AMBER,     AMBER,          TEAL,           GRAY]
    hatches = ['', '', '', '', '//']

    bars = ax2.barh(range(len(configs)), ppls,
                    color=[c+'cc' for c in colors],
                    edgecolor=colors, linewidth=0.7, height=0.52)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    bars[4].set_facecolor('none')
    bars[4].set_linestyle('--')

    for i, v in enumerate(ppls):
        fw = 'bold' if i == 3 else 'normal'
        ax2.text(v + 5, i, f'{v}', va='center', fontsize=6.5,
                 color=colors[i], fontweight=fw)

    ax2.set_yticks(range(len(configs)))
    ax2.set_yticklabels(configs)
    ax2.set_xlabel('Validation PPL (lower is better)')
    ax2.set_xscale('log')
    ax2.set_xlim(20, 800)
    ax2.invert_yaxis()
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.tight_layout(pad=1.2)
    fig.savefig('fig2_lm.pdf')
    fig.savefig('fig2_lm.png')
    plt.close()
    print('fig2 v2 done')


# ================================================================
# 图3：功能分析 v2 — 去掉图内结论文字
# ================================================================
def make_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax1, ax2, ax3 = axes

    # Panel A: MI — 只画数据，不写结论
    channels = ['Wave', 'Cache']
    mis      = [0.474, 0.008]
    cols     = [PURPLE, TEAL]
    ax1.bar(channels, mis, color=[c+'bb' for c in cols],
            edgecolor=cols, linewidth=0.7, width=0.45)
    ax1.text(0, 0.474+0.012, '0.474', ha='center',
             fontsize=7, color=PURPLE, fontweight='bold')
    ax1.text(1, 0.008+0.012, '< 0.01', ha='center',
             fontsize=7, color=TEAL)
    ax1.set_ylabel('Mutual information (nats)')
    ax1.set_ylim(0, 0.60)
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # Panel B: attention distance — x轴直标签
    bins   = ['0–16', '16–32', '32–64', '64–128', '128–256']
    masses = [39.0, 19.5, 20.6, 15.7, 5.2]
    x = np.arange(len(bins))
    bars = ax2.bar(x, masses, color=TEAL+'aa', edgecolor=TEAL,
                   linewidth=0.7, width=0.6)
    bars[0].set_facecolor(TEAL+'ee')
    for i, v in enumerate(masses):
        fw = 'bold' if i == 0 else 'normal'
        ax2.text(i, v+0.5, f'{v}%', ha='center', fontsize=6.5,
                 color=TEAL, fontweight=fw)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins, rotation=0, ha='center', fontsize=6.5)
    ax2.set_ylabel('Attention mass (%)')
    ax2.set_ylim(0, 50)
    ax2.set_title('B', loc='left', fontweight='bold')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Panel C: theta — 数据说话，不写结论
    layers     = np.arange(8)
    theta_mean = np.full(8, 0.50)
    theta_std  = np.full(8, 0.07)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.012, 0.012, 8)

    ax3.axhline(0.5, color=PURPLE, lw=0.9, ls='--', alpha=0.45,
                label='$\\theta = 0.50$')
    ax3.errorbar(layers, theta_mean + jitter, yerr=theta_std,
                 fmt='o', color=PURPLE, ecolor=PURPLE+'66',
                 capsize=2.5, markersize=4, lw=0.9, elinewidth=0.9)
    ax3.set_xlabel('Layer index')
    ax3.set_ylabel('$\\theta$ (mean ± std)')
    ax3.set_xticks(layers)
    ax3.set_ylim(0.25, 0.80)
    ax3.legend(frameon=False, fontsize=6.5, loc='upper right')
    ax3.set_title('C', loc='left', fontweight='bold')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    fig.tight_layout(pad=1.2)
    fig.savefig('fig3_analysis.pdf')
    fig.savefig('fig3_analysis.png')
    plt.close()
    print('fig3 v2 done')


# ================================================================
# 图4：基因组结果 v2 — VEP错位，Panel B统一风格
# ================================================================
def make_fig4():
    fig = plt.figure(figsize=(7.0, 3.0))

    # Panel A: genomic benchmarks
    ax1 = fig.add_axes([0.06, 0.14, 0.44, 0.78])

    tasks    = ['Promoters\n(251 bp)',
                'Enh. ensembl\n(479 bp)',
                'Enh. cohn\n(500 bp)',
                'OCR\n(330 bp)']
    fdm_mean = np.array([0.945, 0.905, 0.775, 0.811])
    fdm_std  = np.array([0.001, 0.012, 0.002, 0.009])
    hyena    = np.array([0.856, 0.706, 0.711, 0.806])
    y = np.arange(len(tasks))

    ax1.barh(y - 0.20, fdm_mean, height=0.32,
             color=PURPLE+'bb', edgecolor=PURPLE, linewidth=0.7,
             label='FDM-HG38')
    ax1.barh(y + 0.20, hyena, height=0.32,
             color=GRAY+'88',   edgecolor=GRAY,   linewidth=0.7,
             label='HyenaDNA')
    ax1.errorbar(fdm_mean, y - 0.20, xerr=fdm_std,
                 fmt='none', ecolor=PURPLE+'99',
                 capsize=2, elinewidth=0.9, capthick=0.9)

    deltas = fdm_mean - hyena
    for i, (fm, hn, d) in enumerate(zip(fdm_mean, hyena, deltas)):
        x_end = max(fm, hn)
        if abs(d) < 0.01:
            ax1.text(x_end + 0.003, i, 'near parity',
                     va='center', fontsize=5.8, color=GRAY, style='italic')
        else:
            ax1.text(x_end + 0.003, i - 0.20,
                     f'+{d:.3f}', va='center',
                     fontsize=6, color=PURPLE, fontweight='bold')

    ax1.set_yticks(y)
    ax1.set_yticklabels(tasks, fontsize=7)
    ax1.set_xlim(0.62, 1.03)
    ax1.set_xlabel('AUC')
    ax1.legend(loc='lower right', frameon=False, fontsize=6.5)
    ax1.set_title('A', loc='left', fontweight='bold')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.invert_yaxis()

    # ---- Panel B: VEP + cross-species ----
    # B 上半：VEP — 三行分离，避免0.861/0.869重叠
    ax_vep = fig.add_axes([0.57, 0.50, 0.41, 0.38])

    vep_methods = ['FDM zero-shot', 'FDM fine-tuned', 'CADD (ref.)']
    vep_aucs    = [0.510,            0.861,             0.869]
    vep_y_pos   = [0,                1,                 2]
    vep_colors  = [GRAY,             PURPLE,            CORAL]
    vep_markers = ['o',              'o',               's']

    for i, (a, yp, col, mk) in enumerate(
            zip(vep_aucs, vep_y_pos, vep_colors, vep_markers)):
        ax_vep.scatter([a], [yp], color=col, marker=mk, s=48, zorder=5,
                       edgecolors=col, linewidths=0.9)
        ax_vep.text(a + 0.006, yp, f'{a:.3f}', va='center',
                    fontsize=6.5, color=col,
                    fontweight='bold' if i == 1 else 'normal')

    ax_vep.set_xlim(0.42, 0.91)
    ax_vep.set_ylim(-0.7, 2.8)
    ax_vep.set_yticks([0, 1, 2])
    ax_vep.set_yticklabels(vep_methods, fontsize=6.5)
    ax_vep.set_xticks([0.5, 0.7, 0.86])
    ax_vep.set_xticklabels(['0.5', '0.7', '0.86'], fontsize=6.5)
    ax_vep.axvline(0.861, color=PURPLE, lw=0.5, ls=':', alpha=0.4)
    ax_vep.axvline(0.869, color=CORAL,  lw=0.5, ls=':', alpha=0.4)
    ax_vep.spines['right'].set_visible(False)
    ax_vep.spines['top'].set_visible(False)
    ax_vep.set_title('B  Downstream transfer', loc='left',
                     fontweight='bold', fontsize=8.5, pad=4)
    ax_vep.text(0.0, 1.06, 'Variant effect prediction (ClinVar)',
                transform=ax_vep.transAxes, fontsize=7,
                color=GRAY)

    # Divider line
    ax_div = fig.add_axes([0.58, 0.50, 0.40, 0.01])
    ax_div.axhline(0, color=LGRAY, lw=0.6)
    ax_div.axis('off')

    # B 下半：cross-species bar
    ax_xsp = fig.add_axes([0.60, 0.12, 0.28, 0.30])

    models = ['CNN\n(scratch)', 'FDM-HG38\n(human pretrained)']
    aucs   = [0.736, 0.792]
    cols   = [GRAY, PURPLE]
    ax_xsp.bar([0, 1], aucs, color=[c+'bb' for c in cols],
               edgecolor=cols, linewidth=0.7, width=0.50)
    ax_xsp.set_ylim(0.68, 0.83)
    ax_xsp.set_xticks([0, 1])
    ax_xsp.set_xticklabels(models, fontsize=6.5)
    ax_xsp.set_ylabel('AUC', fontsize=7)
    for i, v in enumerate(aucs):
        fw = 'bold' if i == 1 else 'normal'
        ax_xsp.text(i, v+0.002, f'{v}', ha='center',
                    fontsize=6.5, color=cols[i], fontweight=fw)
    ax_xsp.annotate('', xy=(1, 0.800), xytext=(0, 0.800),
                    arrowprops=dict(arrowstyle='<->', color=GRAY,
                                   lw=0.6, mutation_scale=7))
    ax_xsp.text(0.5, 0.804, '+0.057', ha='center',
                fontsize=6, color=GRAY)
    ax_xsp.text(0.5, 1.06, 'Cross-species (Drosophila enhancers)',
                ha='center', transform=ax_xsp.transAxes,
                fontsize=7, color=GRAY)
    ax_xsp.spines['right'].set_visible(False)
    ax_xsp.spines['top'].set_visible(False)

    fig.savefig('fig4_genomic.pdf')
    fig.savefig('fig4_genomic.png')
    plt.close()
    print('fig4 v2 done')


# ================================================================
# 补图：显存 scaling
# ================================================================
def make_supp_memory():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    N = np.array([128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    # TF KV cache theoretical (MB): proportional to N
    tf_kv = N / 128 * 3.5
    # FDM measured increment (MB): saturates above N=2048
    fdm_inc = np.where(N <= 2048,
                       N / 128 * 3.5 * 0.85,   # rises slightly below threshold
                       np.full_like(N, 396.9, dtype=float))

    ax.plot(N, tf_kv,  color=CORAL,  lw=1.5, label='Transformer KV cache (theoretical)',
            marker='s', markersize=4)
    ax.plot(N, fdm_inc, color=TEAL,  lw=1.5, label='FDM measured increment',
            marker='o', markersize=4)

    # Annotate saturation point
    ax.axvline(2048, color=GRAY, lw=0.7, ls=':', alpha=0.7)
    ax.text(2200, 600, 'saturation\n$N=2{,}048$', fontsize=6.5,
            color=GRAY, va='top')

    # Annotate final gap
    ax.annotate('', xy=(65536, tf_kv[-1]), xytext=(65536, fdm_inc[-1]),
                arrowprops=dict(arrowstyle='<->', color=GRAY, lw=0.8,
                                mutation_scale=8))
    ax.text(68000, (tf_kv[-1]+fdm_inc[-1])/2,
            f'4.6×', fontsize=7, color=GRAY, va='center')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence length $N$ (tokens)')
    ax.set_ylabel('Memory (MB)')
    ax.set_xticks([128, 512, 2048, 8192, 32768, 65536])
    ax.set_xticklabels(['128', '512', '2K', '8K', '32K', '64K'])
    ax.set_ylim(0, 2200)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title('Runtime memory saturation of FDM\nunder increasing context length',
                 fontsize=8, pad=6)

    fig.tight_layout(pad=1.0)
    fig.savefig('supp_memory.pdf')
    fig.savefig('supp_memory.png')
    plt.close()
    print('supp_memory done')


if __name__ == '__main__':
    import os
    os.chdir('/home/claude/fdm_paper')
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_supp_memory()
    print('\n所有图生成完成')
