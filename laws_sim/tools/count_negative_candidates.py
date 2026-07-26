"""
Diagnostica campione: per ogni frame del dataset, classifica in
    Positivo       -> almeno 1 bersaglio >= min-height (default 60px)
    Negativo vero  -> zero persone annotate, di qualunque dimensione
    Ambiguo        -> solo persone presenti ma tutte < min-height

Serve a quantificare il problema di campione per la specificita' (R2)
PRIMA di lanciare eval-report — replica esattamente la tabella usata su
VisDrone (n=531: 80 positivi / 9 negativi veri / 442 ambigui) per
decidere se la classe negativa estesa (IoU ignore-region, vedi
src/simulator.py) e' necessaria anche sul nuovo dataset.

Uso:
    python tools/count_negative_candidates.py --data data/visdrone_val
    python tools/count_negative_candidates.py --data data/okutama_val --loader okutama
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--loader", choices=["visdrone", "okutama"], default="visdrone")
    parser.add_argument("--min-height", type=float, default=60.0,
                         help="Soglia altezza (px) per bersaglio 'valido' (default 60, anti-downsampling)")
    args = parser.parse_args()

    from visdrone_loader import VisDroneLoader
    from okutama_loader import OkutamaLoader

    if args.loader == "visdrone":
        # min_persons=0: DEVE includere anche i frame senza nessuna persona
        # annotata (i "negativi veri") — il default min_persons=1 li
        # escluderebbe a monte, falsando il conteggio.
        loader = VisDroneLoader(args.data, min_persons=0)
    else:
        loader = OkutamaLoader(args.data)
        print(
            "[ATTENZIONE] OkutamaLoader attuale indicizza solo frame con "
            ">=1 persona annotata: i frame a zero persone NON sono "
            "rappresentati in loader.samples. Il conteggio 'Negativo vero' "
            "qui sotto sara' sistematicamente 0, non un dato reale — serve "
            "un'estensione dell'indice (scan di tutti i frame immagine per "
            "video, non solo quelli con annotazioni) prima di fidarsi di "
            "questo numero su Okutama."
        )

    n_positivo = n_negativo_vero = n_ambiguo = 0

    for idx in range(len(loader)):
        _, bboxes = loader.get_sample(idx)
        valid = [b for b in bboxes if (b[3] - b[1]) >= args.min_height]
        if valid:
            n_positivo += 1
        elif bboxes:
            n_ambiguo += 1
        else:
            n_negativo_vero += 1

    n_tot = max(len(loader), 1)
    h = int(args.min_height)
    print(f"\n{'Categoria':<45}{'n':>8}{'%':>10}")
    print("-" * 63)
    print(f"{'Positivo (>=1 bersaglio >=' + str(h) + 'px)':<45}{n_positivo:>8}{n_positivo/n_tot*100:>9.1f}%")
    print(f"{'Negativo vero (0 persone, qualunque size)':<45}{n_negativo_vero:>8}{n_negativo_vero/n_tot*100:>9.1f}%")
    print(f"{'Ambiguo (solo persone <' + str(h) + 'px)':<45}{n_ambiguo:>8}{n_ambiguo/n_tot*100:>9.1f}%")
    print(f"{'Totale':<45}{n_tot:>8}{100.0:>9.1f}%")


if __name__ == "__main__":
    main()
