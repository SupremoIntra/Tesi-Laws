Eccoci nel vero laboratorio di ricerca. Mettiamo da parte i giocattoli e iniziamo a costruire la pipeline che porterai al professor Rivolta. 

L'obiettivo di questa Task 1 è capire come caricare un'immagine dal disco e trasformarla nel formato matematico esatto che PyTorch (e YOLO) pretende di ricevere prima di subire un attacco.

### 1. Setup dell'Ambiente Pulito

1. Vai nella tua cartella locale (quella sicura, fuori da OneDrive/MEGA) e crea una nuova cartella chiamata **`VisDrone_Experiment`**.
2. Cerca su Google Immagini una foto qualsiasi con vista dall'alto (cerca "VisDrone dataset sample" o "drone view traffic"). Salvala dentro la cartella chiamandola **`01_drone.jpg`**.
3. Apri VS Code in questa cartella (`code .` dal terminale come abbiamo visto) e crea un file chiamato **`task1_dataloader.py`**.

---

### 2. Da OpenCV a PyTorch (La Metamorfosi del Tensore)

Fino ad ora hai visto che OpenCV carica le immagini come un blocco di pixel. Ma le reti neurali come YOLO non ragionano così. Loro vogliono **Tensori Normalizzati**. 

Copia questo codice nel tuo file. Analizzalo bene, perché questa è l'infrastruttura base su cui poi inietteremo il rumore.

```python
import cv2
import torch
from torchvision import transforms

# 1. LETTURA IMMAGINE GREZZA (Il mondo OpenCV)
# OpenCV legge l'immagine come matrice NumPy in formato BGR (Blu, Verde, Rosso)
# e con la struttura [Altezza, Larghezza, Canali]
percorso_img = "01_drone.jpg"
img_cv2 = cv2.imread(percorso_img)

if img_cv2 is None:
    print(f"Errore: Immagine non trovata al percorso {percorso_img}")
    exit()

# 2. PREPARAZIONE COLORI
# PyTorch e YOLO usano lo standard RGB (Rosso, Verde, Blu), quindi scambiamo i canali
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

# 3. LA TRASFORMAZIONE IN TENSORE (Il mondo PyTorch)
# transforms.ToTensor() è un comando magico che fa due cose fondamentali:
# - Cambia la struttura da [Altezza, Larghezza, Canali] a [Canali, Altezza, Larghezza]
# - Schiaccia il valore dei pixel da (0-255) a decimali (0.0 - 1.0)
trasforma_in_tensore = transforms.ToTensor()
img_tensore = trasforma_in_tensore(img_rgb)

# 4. SPOSTAMENTO SULLA GPU DEL MAC (L'hardware)
# Spostiamo la matrice matematica sul chip M4 per i calcoli futuri
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
img_tensore = img_tensore.to(device)

# --- STAMPIAMO I RISULTATI DELLA METAMORFOSI ---
print("\n--- ANALISI DEI DATI ---")
print(f"1. Forma originale OpenCV (H, W, C): {img_cv2.shape}")
print(f"2. Forma Tensore PyTorch (C, H, W): {img_tensore.shape}")
print(f"3. Valore di un pixel a caso (prima): {img_cv2[0, 0, 0]} (Intero 0-255)")
print(f"4. Valore di un pixel a caso (dopo): {img_tensore[0, 0, 0]:.4f} (Float 0.0-1.0)")
print(f"5. Hardware attivo: {img_tensore.device}\n")
```

---

### 3. I Concetti Chiave da "Vendere" al Professore

Quando farai girare questo codice dal tuo terminale (`python task1_dataloader.py`), l'output ti stamperà la radiografia esatta dell'immagine.

* **La regola del "Channel First" (`C, H, W`):** Noterai che la dimensione del colore (es. `3`) nel tensore PyTorch viene spostata all'inizio. I programmatori classici tengono il colore alla fine, ma i ricercatori di Deep Learning lo mettono all'inizio perché rende le moltiplicazioni di matrici (le convoluzioni di cui parlavi nel corso PyTorch) molto più veloci.
* **La Normalizzazione (`0.0 - 1.0`):** I pixel originali vanno da 0 a 255. Se passassi questi numeri giganti a una rete neurale, i calcoli dei gradienti (la derivata dell'errore) "esploderebbero". Portando tutto tra `0.0` e `1.0`, la matematica del modello rimane stabile.

Questo script è il tuo ponte. Hai preso i dati dal mondo reale e li hai formattati esattamente come li vuole il "cervello" della macchina, posizionandoli sulla memoria unificata del tuo Mac M4. L'immagine ora è un tensore pronto per essere manipolato, ritagliato o, nel tuo caso, "avvelenato".