import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os, math


C_BG      = "#0D1117"
C_SURFACE = "#161B22"
C_PANEL   = "#1C2128"
C_BORDER  = "#30363D"
C_BTN     = "#21262D"
C_BTN_HV  = "#30363D"
C_ACCENT  = "#58A6FF"
C_GREEN   = "#3FB950"
C_ORANGE  = "#D29922"
C_RED     = "#F85149"
C_PURPLE  = "#BC8CFF"
C_PINK    = "#FF7EB6"
C_TXT     = "#E6EDF3"
C_DIM     = "#8B949E"

FT = ("Consolas", 9)
FT_B = ("Consolas", 9, "bold")
FT_H = ("Consolas", 11, "bold")
FT_T = ("Consolas", 13, "bold")



def to_gray(img):
    if img.ndim == 2: return img
    r,g,b = img[:,:,0].astype(np.float64), img[:,:,1].astype(np.float64), img[:,:,2].astype(np.float64)
    return np.clip(0.299*r + 0.587*g + 0.114*b, 0, 255).astype(np.uint8)



def contrast_brightness(img, alpha=1.0, beta=0):
    
    g = img.astype(np.float64)
    return np.clip(alpha * g + beta, 0, 255).astype(np.uint8)

def compute_hist(gray):
    h = np.zeros(256, dtype=np.int64)
    flat = gray.flatten()
    for v in flat:
        h[v] += 1
    return h

def hist_equalize(gray):
    h = compute_hist(gray)
    cdf = np.cumsum(h)
    cdf_min = cdf[cdf > 0][0]
    n = gray.size
    lut = np.clip(((cdf - cdf_min) / (n - cdf_min) * 255), 0, 255).astype(np.uint8)
    return lut[gray]

def otsu_threshold(gray):
    h = compute_hist(gray)
    total = gray.size
    sum_total = float(np.sum(np.arange(256, dtype=np.float64) * h))
    sum_b, w_b, var_max, thresh = 0.0, 0, 0.0, 0
    for t in range(256):
        w_b += h[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * h[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > var_max:
            var_max = var; thresh = t
    return (gray >= thresh).astype(np.uint8) * 255, thresh



def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    pad = np.pad(img.astype(np.float64), ((ph,ph),(pw,pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.sum(pad[i:i+kh, j:j+kw] * kernel)
    return out

def mean_filter(gray, k=3):
    kernel = np.ones((k,k)) / (k*k)
    return np.clip(convolve2d(gray, kernel), 0, 255).astype(np.uint8)

def gaussian_kernel(k, sigma=1.0):
    ax = np.arange(-(k//2), k//2+1)
    xx, yy = np.meshgrid(ax, ax)
    kern = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    return kern / kern.sum()

def gaussian_filter(gray, k=3, sigma=1.0):
    return np.clip(convolve2d(gray, gaussian_kernel(k, sigma)), 0, 255).astype(np.uint8)

def median_filter(gray, k=3):
    ph = k//2
    pad = np.pad(gray.astype(np.float64), ph, mode="reflect")
    out = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            window = pad[i:i+k, j:j+k].flatten()
            out[i,j] = np.median(window)
    return out.astype(np.uint8)

def sobel_filter(gray):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)
    gx = convolve2d(gray, kx); gy = convolve2d(gray, ky)
    return np.clip(np.sqrt(gx**2+gy**2), 0, 255).astype(np.uint8)

def prewitt_filter(gray):
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float64)
    ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float64)
    gx = convolve2d(gray, kx); gy = convolve2d(gray, ky)
    return np.clip(np.sqrt(gx**2+gy**2), 0, 255).astype(np.uint8)

def laplacian_filter(gray):
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
    out = convolve2d(gray, k)
    return np.clip(np.abs(out), 0, 255).astype(np.uint8)

def unsharp_mask(gray, k=3, sigma=1.0, strength=1.5):
    blurred = gaussian_filter(gray, k, sigma).astype(np.float64)
    sharp = gray.astype(np.float64) + strength * (gray.astype(np.float64) - blurred)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _struct_elem(k):
    return np.ones((k,k), dtype=np.uint8)

def erode(binary, k=3):
    se = _struct_elem(k); ph = k//2
    pad = np.pad(binary, ph, mode="constant", constant_values=255)
    out = np.zeros_like(binary)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = pad[i:i+k, j:j+k]
            out[i,j] = 255 if np.all(region == 255) else 0
    return out

def dilate(binary, k=3):
    se = _struct_elem(k); ph = k//2
    pad = np.pad(binary, ph, mode="constant", constant_values=0)
    out = np.zeros_like(binary)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = pad[i:i+k, j:j+k]
            out[i,j] = 255 if np.any(region == 255) else 0
    return out

def opening(binary, k=3):
    return dilate(erode(binary, k), k)

def closing(binary, k=3):
    return erode(dilate(binary, k), k)

def zhang_suen_skeleton(binary):
    
    img = (binary > 127).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in [1, 2]:
            to_remove = []
            rows, cols = img.shape
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if img[i,j] != 1: continue
                    p = [img[i-1,j], img[i-1,j+1], img[i,j+1], img[i+1,j+1],
                         img[i+1,j], img[i+1,j-1], img[i,j-1], img[i-1,j-1]]
                    B = sum(p)
                    if not (2 <= B <= 6): continue
                    A = sum(1 for k in range(8) if p[k]==0 and p[(k+1)%8]==1)
                    if A != 1: continue
                    if step == 1:
                        if p[0]*p[2]*p[4] != 0: continue
                        if p[2]*p[4]*p[6] != 0: continue
                    else:
                        if p[0]*p[2]*p[6] != 0: continue
                        if p[0]*p[4]*p[6] != 0: continue
                    to_remove.append((i,j))
            for (i,j) in to_remove:
                img[i,j] = 0; changed = True
    return (img * 255).astype(np.uint8)



def styled_btn(parent, text, cmd, color=C_ACCENT, width=None):
    f = tk.Frame(parent, bg=color, padx=1, pady=1)
    kw = {}
    if width: kw["width"] = width
    b = tk.Button(f, text=text, command=cmd, font=FT_B,
                  bg=C_BTN, fg=C_TXT, relief=tk.FLAT, bd=0,
                  activebackground=color, activeforeground=C_TXT,
                  padx=6, pady=5, cursor="hand2", **kw)
    b.pack(fill=tk.X)
    b.bind("<Enter>", lambda e: b.config(bg=color))
    b.bind("<Leave>", lambda e: b.config(bg=C_BTN))
    return f, b

def section_label(parent, text, color=C_ACCENT):
    tk.Label(parent, text=f"  {text}", font=FT_H, bg=C_SURFACE,
             fg=color, anchor="w").pack(fill=tk.X, pady=(10,3))

def sep(parent):
    tk.Frame(parent, bg=C_BORDER, height=1).pack(fill=tk.X, pady=4)



CANVAS_W, CANVAS_H = 2000, 1500

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Custom-IPT  ·  Image Processing Toolbox")
        self.root.configure(bg=C_BG)
        self.root.geometry("1380x820")
        self.root.minsize(1100, 700)

        self.history    = []
        self.redo_stack = []
        self.orig_rgb   = None
        self.img_path   = None

        self.tool_mode  = None   
        self.click_pts  = []
        self._line_id   = None

        self._build_ui()
        self._bind_keys()



    def _build_ui(self):

        menubar = tk.Menu(self.root, bg=C_SURFACE, fg=C_TXT,
                          activebackground=C_ACCENT, activeforeground=C_BG,
                          relief=tk.FLAT)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, bg=C_SURFACE, fg=C_TXT,
                            activebackground=C_ACCENT, activeforeground=C_BG)
        menubar.add_cascade(label="  Fichier  ", menu=file_menu)
        file_menu.add_command(label="Ouvrir…",          command=self.open_img,  accelerator="Ctrl+O")
        file_menu.add_command(label="Enregistrer sous…", command=self.save_img, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Reset image originale", command=self.reset_img)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0, bg=C_SURFACE, fg=C_TXT,
                            activebackground=C_ACCENT, activeforeground=C_BG)
        menubar.add_cascade(label="  Édition  ", menu=edit_menu)
        edit_menu.add_command(label="Annuler (Undo)", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Rétablir (Redo)", command=self.redo, accelerator="Ctrl+Y")


        topbar = tk.Frame(self.root, bg=C_BG, pady=8)
        topbar.pack(fill=tk.X, padx=16)
        tk.Label(topbar, text="◈ Custom-IPT", font=("Consolas",14,"bold"),
                 bg=C_BG, fg=C_ACCENT).pack(side=tk.LEFT)
        tk.Label(topbar, text="Image Processing Toolbox · Ingénieur 4 IA",
                 font=FT, bg=C_BG, fg=C_DIM).pack(side=tk.LEFT, padx=12)
        self.info_lbl = tk.Label(topbar, text="Aucune image", font=FT,
                                 bg=C_BG, fg=C_DIM)
        self.info_lbl.pack(side=tk.RIGHT)


        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0,8))

        
        left = tk.Frame(body, bg=C_SURFACE, width=500,
                        highlightbackground=C_BORDER, highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        self._build_sidebar(left)

        
        center = tk.Frame(body, bg=C_BG)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self._build_canvas(center)

        
        right = tk.Frame(body, bg=C_SURFACE, width=600,
                         highlightbackground=C_BORDER, highlightthickness=1)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        self._build_right(right)


        sb = tk.Frame(self.root, bg=C_PANEL, height=24)
        sb.pack(fill=tk.X)
        self.status = tk.Label(sb, text="  Prêt. Ouvrez une image (Fichier → Ouvrir).",
                               font=FT, bg=C_PANEL, fg=C_DIM, anchor="w")
        self.status.pack(side=tk.LEFT, fill=tk.X)
        self.hist_info = tk.Label(sb, text="", font=FT, bg=C_PANEL, fg=C_DIM)
        self.hist_info.pack(side=tk.RIGHT, padx=10)



    def _build_sidebar(self, parent):
        
        canvas = tk.Canvas(parent, bg=C_SURFACE, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(canvas, bg=C_SURFACE)
        canvas.create_window((0,0), window=self.sidebar, anchor="nw")
        self.sidebar.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta>0 else 1, "units"))

        s = self.sidebar

        
        section_label(s, "FICHIER", C_ACCENT)
        for txt, cmd, col in [(" Ouvrir", self.open_img, C_ACCENT),
                               (" Enregistrer", self.save_img, C_GREEN),
                               ("↺  Reset original", self.reset_img, C_RED)]:
            f,_ = styled_btn(s, txt, cmd, col); f.pack(fill=tk.X, padx=8, pady=2)

        
        sep(s)
        row = tk.Frame(s, bg=C_SURFACE); row.pack(fill=tk.X, padx=8, pady=2)
        f1,_ = styled_btn(row, "◀ Undo", self.undo, C_PINK); f1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(row, bg=C_SURFACE, width=4).pack(side=tk.LEFT)
        f2,_ = styled_btn(row, "Redo ▶", self.redo, C_GREEN); f2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.step_lbl = tk.Label(s, text="", font=FT, bg=C_SURFACE, fg=C_DIM)
        self.step_lbl.pack(pady=2)


        sep(s)
        section_label(s, "A · POINT-TO-POINT", C_ORANGE)

        # Contrast & Brightness
        tk.Label(s, text="  Contraste α  (×)", font=FT, bg=C_SURFACE, fg=C_DIM, anchor="w").pack(fill=tk.X)
        self.alpha_var = tk.DoubleVar(value=1.0)
        tk.Scale(s, from_=0.1, to=3.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.alpha_var, bg=C_SURFACE, fg=C_TXT, troughcolor=C_BTN,
                 highlightthickness=0, sliderlength=12, length=200).pack(padx=10)

        tk.Label(s, text="  Luminosité β  (+/-)", font=FT, bg=C_SURFACE, fg=C_DIM, anchor="w").pack(fill=tk.X)
        self.beta_var = tk.IntVar(value=0)
        tk.Scale(s, from_=-128, to=128, resolution=1, orient=tk.HORIZONTAL,
                 variable=self.beta_var, bg=C_SURFACE, fg=C_TXT, troughcolor=C_BTN,
                 highlightthickness=0, sliderlength=12, length=200).pack(padx=10)

        f,_ = styled_btn(s, "Appliquer g=αf+β", self.do_contrast, C_ORANGE)
        f.pack(fill=tk.X, padx=8, pady=3)

        sep(s)
        f,_ = styled_btn(s, "Égalisation histogramme", self.do_histeq, C_ORANGE)
        f.pack(fill=tk.X, padx=8, pady=2)

        tk.Label(s, text="  Seuillage", font=FT, bg=C_SURFACE, fg=C_DIM, anchor="w").pack(fill=tk.X)
        row2 = tk.Frame(s, bg=C_SURFACE); row2.pack(fill=tk.X, padx=8, pady=2)
        f1,_ = styled_btn(row2, "Otsu (auto)", self.do_otsu, C_ORANGE)
        f1.pack(side=tk.LEFT, fill=tk.X, expand=True)


        sep(s)
        section_label(s, "B · FILTRAGE SPATIAL", C_PURPLE)

        tk.Label(s, text="  Taille noyau", font=FT, bg=C_SURFACE, fg=C_DIM, anchor="w").pack(fill=tk.X)
        self.kernel_var = tk.IntVar(value=3)
        krow = tk.Frame(s, bg=C_SURFACE); krow.pack(pady=2)
        for v in [3,5,7]:
            tk.Radiobutton(krow, text=f"{v}×{v}", variable=self.kernel_var, value=v,
                           bg=C_SURFACE, fg=C_TXT, selectcolor=C_BTN,
                           activebackground=C_SURFACE, font=FT).pack(side=tk.LEFT)

        tk.Label(s, text="  σ Gaussien", font=FT, bg=C_SURFACE, fg=C_DIM, anchor="w").pack(fill=tk.X)
        self.sigma_var = tk.DoubleVar(value=1.0)
        tk.Scale(s, from_=0.3, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=self.sigma_var, bg=C_SURFACE, fg=C_TXT, troughcolor=C_BTN,
                 highlightthickness=0, sliderlength=12, length=200).pack(padx=10)

        smoothing = [("Filtre Moyenneur",  self.do_mean,     C_PURPLE),
                     ("Filtre Médian",     self.do_median,   C_PURPLE),
                     ("Filtre Gaussien",   self.do_gauss,    C_PURPLE)]
        for txt, cmd, col in smoothing:
            f,_ = styled_btn(s, txt, cmd, col); f.pack(fill=tk.X, padx=8, pady=2)

        sep(s)
        edges = [("Sobel",            self.do_sobel,   C_PINK),
                 ("Prewitt",          self.do_prewitt, C_PINK),
                 ("Laplacien",        self.do_laplace, C_PINK),
                 ("Netteté (Unsharp)",self.do_unsharp, C_PINK)]
        for txt, cmd, col in edges:
            f,_ = styled_btn(s, txt, cmd, col); f.pack(fill=tk.X, padx=8, pady=2)


        sep(s)
        section_label(s, "C · MORPHOLOGIE", C_GREEN)

        tk.Label(s, text="  Taille SE (structurant)", font=FT, bg=C_SURFACE, fg=C_DIM, anchor="w").pack(fill=tk.X)
        self.morph_k = tk.IntVar(value=3)
        mrow = tk.Frame(s, bg=C_SURFACE); mrow.pack(pady=2)
        for v in [3,5,7]:
            tk.Radiobutton(mrow, text=f"{v}×{v}", variable=self.morph_k, value=v,
                           bg=C_SURFACE, fg=C_TXT, selectcolor=C_BTN,
                           activebackground=C_SURFACE, font=FT).pack(side=tk.LEFT)

        morph_ops = [("Érosion",   self.do_erode,    C_GREEN),
                     ("Dilatation", self.do_dilate,   C_GREEN),
                     ("Ouverture",  self.do_opening,  C_GREEN),
                     ("Fermeture",  self.do_closing,  C_GREEN),
                     ("Squelette", self.do_skeleton, C_GREEN)]
        for txt, cmd, col in morph_ops:
            f,_ = styled_btn(s, txt, cmd, col); f.pack(fill=tk.X, padx=8, pady=2)


        sep(s)
        section_label(s, "ANALYSE", C_ACCENT)

        analysis = [("↔  Profil de ligne", self.tool_line_profile, C_ACCENT),
                    (" Mesure distance",  self.tool_distance,     C_ACCENT),
                    ("  Désactiver outil", self.disable_tool,     C_DIM)]
        for txt, cmd, col in analysis:
            f,_ = styled_btn(s, txt, cmd, col); f.pack(fill=tk.X, padx=8, pady=2)

        self.tool_lbl = tk.Label(s, text="", font=FT, bg=C_SURFACE, fg=C_DIM, wraplength=200)
        self.tool_lbl.pack(padx=8, pady=4)

        # Grayscale
        sep(s)
        f,_ = styled_btn(s, "⬛ Convertir en Niveaux de Gris", self.do_gray, C_DIM)
        f.pack(fill=tk.X, padx=8, pady=2)
        tk.Frame(s, bg=C_SURFACE, height=10).pack()



    def _build_canvas(self, parent):
        container = tk.Frame(parent, bg=C_PANEL,
                             highlightbackground=C_BORDER, highlightthickness=1)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg="#050508", width=CANVAS_W, height=CANVAS_H,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas.create_text(CANVAS_W//2, CANVAS_H//2,
            text="Ouvrez une image pour commencer",
            fill=C_DIM, font=("Consolas",11), tags="placeholder")

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>",   self._on_canvas_motion)

        self.cursor_lbl = tk.Label(parent, text="", font=FT, bg=C_PANEL, fg=C_DIM)
        self.cursor_lbl.pack()



    def _build_right(self, parent):
        tk.Label(parent, text="  HISTOGRAMME", font=FT_H,
                 bg=C_SURFACE, fg=C_PINK, anchor="w").pack(fill=tk.X, pady=(10,2))

        self.fig_hist, self.ax_hist = plt.subplots(figsize=(4.5, 3.0), dpi=96)
        self.fig_hist.patch.set_facecolor(C_SURFACE)
        self.ax_hist.set_facecolor(C_BG)
        self.ax_hist.tick_params(colors=C_DIM, labelsize=6)
        for sp in self.ax_hist.spines.values(): sp.set_edgecolor(C_BORDER)
        self.fig_hist.tight_layout(pad=0.8)

        self.hc = FigureCanvasTkAgg(self.fig_hist, master=parent)
        self.hc.get_tk_widget().config(bg=C_SURFACE, highlightthickness=0, height=280)
        self.hc.get_tk_widget().pack(fill=tk.X, padx=6)

        sep_frame = tk.Frame(parent, bg=C_BORDER, height=1)
        sep_frame.pack(fill=tk.X, pady=6, padx=6)

        tk.Label(parent, text="  PROFIL / ANALYSE", font=FT_H,
                 bg=C_SURFACE, fg=C_ACCENT, anchor="w").pack(fill=tk.X)

        self.fig_prof, self.ax_prof = plt.subplots(figsize=(2.9, 1.9), dpi=96)
        self.fig_prof.patch.set_facecolor(C_SURFACE)
        self.ax_prof.set_facecolor(C_BG)
        self.ax_prof.tick_params(colors=C_DIM, labelsize=6)
        for sp in self.ax_prof.spines.values(): sp.set_edgecolor(C_BORDER)
        self.fig_prof.tight_layout(pad=0.8)

        self.pc = FigureCanvasTkAgg(self.fig_prof, master=parent)
        self.pc.get_tk_widget().config(bg=C_SURFACE, highlightthickness=0, height=180)
        self.pc.get_tk_widget().pack(fill=tk.X, padx=6)

        sep_frame2 = tk.Frame(parent, bg=C_BORDER, height=1)
        sep_frame2.pack(fill=tk.X, pady=6, padx=6)

        # Stats box
        tk.Label(parent, text="  STATISTIQUES", font=FT_H,
                 bg=C_SURFACE, fg=C_GREEN, anchor="w").pack(fill=tk.X)
        self.stats_lbl = tk.Label(parent, text="—", font=FT, bg=C_SURFACE,
                                  fg=C_DIM, justify=tk.LEFT, anchor="w", padx=10)
        self.stats_lbl.pack(fill=tk.X)

        self.dist_lbl = tk.Label(parent, text="", font=("Consolas",10,"bold"),
                                 bg=C_SURFACE, fg=C_GREEN, padx=10)
        self.dist_lbl.pack(fill=tk.X, pady=4)



    def _on_canvas_motion(self, event):
        if not self.history: return
        arr = self.history[-1]
        x, y = self._canvas_to_img(event.x, event.y, arr)
        if x is None: return
        if arr.ndim == 2:
            v = arr[y, x]
            self.cursor_lbl.config(text=f"  ({x}, {y})  |  I={v}")
        else:
            r,g,b = arr[y,x]
            self.cursor_lbl.config(text=f"  ({x}, {y})  |  R={r} G={g} B={b}")

    def _on_canvas_click(self, event):
        if not self.history: return
        arr = self.history[-1]
        x, y = self._canvas_to_img(event.x, event.y, arr)
        if x is None: return

        if self.tool_mode == "line":
            self.click_pts.append((x, y))
            # draw dot on canvas
            cx, cy = self._img_to_canvas(x, y, arr)
            self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4, fill=C_ACCENT, outline="", tags="overlay")
            if len(self.click_pts) == 2:
                self._draw_line_profile()
                self.click_pts = []

        elif self.tool_mode == "distance":
            self.click_pts.append((x, y))
            cx, cy = self._img_to_canvas(x, y, arr)
            self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4, fill=C_GREEN, outline="", tags="overlay")
            if len(self.click_pts) == 2:
                p1, p2 = self.click_pts
                d = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                cx1,cy1 = self._img_to_canvas(p1[0],p1[1],arr)
                cx2,cy2 = self._img_to_canvas(p2[0],p2[1],arr)
                self.canvas.create_line(cx1,cy1,cx2,cy2, fill=C_GREEN, width=2, tags="overlay")
                self.dist_lbl.config(text=f"  Distance: {d:.1f} px")
                self.tool_lbl.config(text=f"Distance: {d:.1f} px\n({p1[0]},{p1[1]}) → ({p2[0]},{p2[1]})")
                self._set_status(f"Distance mesurée: {d:.1f} pixels")
                self.click_pts = []

    def _canvas_to_img(self, cx, cy, arr):
        h, w = arr.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        nw, nh = int(w*scale), int(h*scale)
        ox = (CANVAS_W - nw) // 2
        oy = (CANVAS_H - nh) // 2
        ix = int((cx - ox) / scale)
        iy = int((cy - oy) / scale)
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None, None

    def _img_to_canvas(self, ix, iy, arr):
        h, w = arr.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        nw, nh = int(w*scale), int(h*scale)
        ox = (CANVAS_W - nw) // 2
        oy = (CANVAS_H - nh) // 2
        return int(ix * scale + ox), int(iy * scale + oy)

    def _draw_line_profile(self):
        arr = self.history[-1]
        gray = to_gray(arr) if arr.ndim == 3 else arr
        p1, p2 = self.click_pts
        length = int(math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
        if length == 0: return
        xs = np.linspace(p1[0], p2[0], length).astype(int)
        ys = np.linspace(p1[1], p2[1], length).astype(int)
        xs = np.clip(xs, 0, gray.shape[1]-1)
        ys = np.clip(ys, 0, gray.shape[0]-1)
        profile = gray[ys, xs]

        
        cx1,cy1 = self._img_to_canvas(p1[0],p1[1],arr)
        cx2,cy2 = self._img_to_canvas(p2[0],p2[1],arr)
        self.canvas.create_line(cx1,cy1,cx2,cy2, fill=C_ACCENT, width=2, dash=(4,2), tags="overlay")

        
        self.ax_prof.clear()
        self.ax_prof.set_facecolor(C_BG)
        self.ax_prof.tick_params(colors=C_DIM, labelsize=6)
        for sp in self.ax_prof.spines.values(): sp.set_edgecolor(C_BORDER)
        self.ax_prof.plot(profile, color=C_ACCENT, linewidth=1)
        self.ax_prof.set_xlabel("Position (px)", color=C_DIM, fontsize=6)
        self.ax_prof.set_ylabel("Intensité", color=C_DIM, fontsize=6)
        self.ax_prof.set_title("Profil de ligne", color=C_TXT, fontsize=7)
        self.fig_prof.tight_layout(pad=0.8)
        self.pc.draw()
        self._set_status(f"Profil de ligne: {length} pixels, I∈[{profile.min()},{profile.max()}]")



    def _show(self, arr):
        if arr.ndim == 2:
            img = Image.fromarray(arr, "L").convert("RGB")
        else:
            img = Image.fromarray(arr)

        h, w = arr.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        nw, nh = int(w*scale), int(h*scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(CANVAS_W//2, CANVAS_H//2, image=self.tkimg)

        self._update_histogram(arr)
        self._update_stats(arr)

        mode = "Niveaux de gris" if arr.ndim == 2 else "Couleur RGB"
        self.info_lbl.config(text=f"{w}×{h} px  |  {mode}")
        self.step_lbl.config(text=f"  Historique: {len(self.history)}   Redo: {len(self.redo_stack)}")

    def _update_histogram(self, arr):
        self.ax_hist.clear()
        self.ax_hist.set_facecolor(C_BG)
        self.ax_hist.tick_params(colors=C_DIM, labelsize=6)
        for sp in self.ax_hist.spines.values(): sp.set_edgecolor(C_BORDER)

        if arr.ndim == 2:
            h = compute_hist(arr)
            self.ax_hist.bar(np.arange(256), h, color=C_PINK, alpha=0.85, width=1, linewidth=0)
        else:
            for ch, col, lbl in [(0,C_RED,"R"),(1,C_GREEN,"G"),(2,C_ACCENT,"B")]:
                h = compute_hist(arr[:,:,ch].astype(np.uint8))
                self.ax_hist.plot(h, color=col, linewidth=0.8, alpha=0.85, label=lbl)
            self.ax_hist.legend(fontsize=6, facecolor=C_PANEL, edgecolor=C_BORDER,
                                labelcolor=C_TXT, loc="upper right")

        self.ax_hist.set_xlim(0, 255)
        self.fig_hist.tight_layout(pad=0.8)
        self.hc.draw()

    def _update_stats(self, arr):
        gray = to_gray(arr) if arr.ndim == 3 else arr
        self.stats_lbl.config(text=(
            f"  Min:    {gray.min()}\n"
            f"  Max:    {gray.max()}\n"
            f"  Moy:    {gray.mean():.1f}\n"
            f"  σ:      {gray.std():.1f}\n"
            f"  Taille: {gray.shape[1]}×{gray.shape[0]}"
        ))



    def _push(self, arr):
        self.history.append(arr.copy())
        self.redo_stack.clear()
        self._show(arr)

    def _current(self):
        return self.history[-1] if self.history else None

    def _get_gray(self):
        cur = self._current()
        if cur is None: return None
        return to_gray(cur) if cur.ndim == 3 else cur

    def _set_status(self, msg):
        self.status.config(text=f"  {msg}")



    def open_img(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),("Tous","*.*")])
        if not path: return
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        self.orig_rgb = arr.copy()
        self.img_path = path
        self.history.clear(); self.redo_stack.clear()
        self._push(arr)
        self._set_status(f"Ouvert: {os.path.basename(path)}")

    def save_img(self):
        cur = self._current()
        if cur is None: messagebox.showwarning("Attention","Aucune image."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp")])
        if not path: return
        if cur.ndim == 2:
            Image.fromarray(cur, "L").save(path)
        else:
            Image.fromarray(cur).save(path)
        self._set_status(f"Enregistré: {os.path.basename(path)}")

    def reset_img(self):
        if self.orig_rgb is None: return
        self.history.clear(); self.redo_stack.clear()
        self._push(self.orig_rgb.copy())
        self.canvas.delete("overlay")
        self.dist_lbl.config(text="")
        self._set_status("Image réinitialisée à l'originale.")

    def undo(self):
        if len(self.history) <= 1: self._set_status("Rien à annuler."); return
        self.redo_stack.append(self.history.pop())
        self._show(self.history[-1])
        self.step_lbl.config(text=f"  Historique: {len(self.history)}   Redo: {len(self.redo_stack)}")
        self._set_status("Annulé (Undo).")

    def redo(self):
        if not self.redo_stack: self._set_status("Rien à rétablir."); return
        arr = self.redo_stack.pop()
        self.history.append(arr)
        self._show(arr)
        self.step_lbl.config(text=f"  Historique: {len(self.history)}   Redo: {len(self.redo_stack)}")
        self._set_status("Rétabli (Redo).")



    def do_gray(self):
        cur = self._current()
        if cur is None: return
        self._push(to_gray(cur)); self._set_status("Converti en niveaux de gris.")

    def do_contrast(self):
        g = self._get_gray();
        if g is None: return
        a, b = self.alpha_var.get(), self.beta_var.get()
        self._push(contrast_brightness(g, a, b))
        self._set_status(f"Contraste/Luminosité: α={a:.2f}, β={b}")

    def do_histeq(self):
        g = self._get_gray();
        if g is None: return
        self._push(hist_equalize(g)); self._set_status("Égalisation d'histogramme appliquée.")

    def do_otsu(self):
        g = self._get_gray();
        if g is None: return
        result, t = otsu_threshold(g)
        self._push(result); self._set_status(f"Seuillage Otsu: seuil = {t}")



    def do_mean(self):
        g = self._get_gray(); k = self.kernel_var.get()
        if g is None: return
        self._push(mean_filter(g, k)); self._set_status(f"Filtre moyenneur {k}×{k}.")

    def do_median(self):
        g = self._get_gray(); k = self.kernel_var.get()
        if g is None: return
        self._set_status(f"Calcul filtre médian {k}×{k}… (peut prendre quelques secondes)")
        self.root.update()
        self._push(median_filter(g, k)); self._set_status(f"Filtre médian {k}×{k} appliqué.")

    def do_gauss(self):
        g = self._get_gray(); k = self.kernel_var.get(); s = self.sigma_var.get()
        if g is None: return
        self._push(gaussian_filter(g, k, s)); self._set_status(f"Filtre Gaussien {k}×{k} σ={s:.1f}.")

    def do_sobel(self):
        g = self._get_gray();
        if g is None: return
        self._push(sobel_filter(g)); self._set_status("Détection de contours Sobel.")

    def do_prewitt(self):
        g = self._get_gray();
        if g is None: return
        self._push(prewitt_filter(g)); self._set_status("Détection de contours Prewitt.")

    def do_laplace(self):
        g = self._get_gray();
        if g is None: return
        self._push(laplacian_filter(g)); self._set_status("Filtre Laplacien appliqué.")

    def do_unsharp(self):
        g = self._get_gray(); k = self.kernel_var.get(); s = self.sigma_var.get()
        if g is None: return
        self._push(unsharp_mask(g, k, s)); self._set_status(f"Unsharp Masking {k}×{k} σ={s:.1f}.")

    def _get_binary(self):
        g = self._get_gray()
        if g is None: return None
        if not np.all((g == 0) | (g == 255)):
            r, _ = otsu_threshold(g)
            self._set_status("Image binarisée automatiquement (Otsu) avant opération morphologique.")
            return r
        return g

    def do_erode(self):
        b = self._get_binary(); k = self.morph_k.get()
        if b is None: return
        self._push(erode(b, k)); self._set_status(f"Érosion {k}×{k}.")

    def do_dilate(self):
        b = self._get_binary(); k = self.morph_k.get()
        if b is None: return
        self._push(dilate(b, k)); self._set_status(f"Dilatation {k}×{k}.")

    def do_opening(self):
        b = self._get_binary(); k = self.morph_k.get()
        if b is None: return
        self._push(opening(b, k)); self._set_status(f"Ouverture {k}×{k}.")

    def do_closing(self):
        b = self._get_binary(); k = self.morph_k.get()
        if b is None: return
        self._push(closing(b, k)); self._set_status(f"Fermeture {k}×{k}.")

    def do_skeleton(self):
        b = self._get_binary()
        if b is None: return
        self._set_status("Calcul squelette Zhang-Suen… (peut prendre un moment)")
        self.root.update()
        self._push(zhang_suen_skeleton(b)); self._set_status("Squelette Zhang-Suen calculé.")

    # ── Outils d'analyse ──────────────────────────────────────────────────────

    def tool_line_profile(self):
        self.tool_mode = "line"
        self.click_pts = []
        self.canvas.delete("overlay")
        self.tool_lbl.config(text="Cliquez 2 points\nsur l'image pour\ntacer le profil.", fg=C_ACCENT)
        self._set_status("Outil Profil de ligne: cliquez 2 points sur l'image.")

    def tool_distance(self):
        self.tool_mode = "distance"
        self.click_pts = []
        self.canvas.delete("overlay")
        self.dist_lbl.config(text="")
        self.tool_lbl.config(text="Cliquez 2 points\npour mesurer la\ndistance.", fg=C_GREEN)
        self._set_status("Outil Distance: cliquez 2 points.")

    def disable_tool(self):
        self.tool_mode = None
        self.click_pts = []
        self.canvas.delete("overlay")
        self.tool_lbl.config(text="Outil désactivé.", fg=C_DIM)
        self._set_status("Outil désactivé.")



    def _bind_keys(self):
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-o>", lambda e: self.open_img())
        self.root.bind("<Control-s>", lambda e: self.save_img())


if __name__ == "__main__":
    root = tk.Tk()
    root.tk_setPalette(background=C_BG, foreground=C_TXT)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Vertical.TScrollbar", background=C_BTN, troughcolor=C_SURFACE,
                    bordercolor=C_BORDER, arrowcolor=C_DIM)

    app = App(root)
    root.mainloop()
