# Integrazione di strategie di Continual Learning e Machine Unlearning per la mitigazione del concept drift in ML-NIDS

Questa repository contiene il framework originale sviluppato per l'omonima tesi magistrale.

Il progetto affronta il decadimento temporale dei ML-NIDS causato principalmente dal **Concept Drift**, e anche da attacchi **Adversarial**.

## Caratteristiche principali

Il framework implementa tre principali caratteristiche:
- **Modular Network**: Architettura del framework. Basata sulla creazione di modelli indipendenti tra loro, che lavorano insieme (ensemble).
- **Experience Replay**: Gestione del Continual Learning tramite un buffer contenente i campioni critici (più vicini al decision boundary) per contrastare il Concept Drift.\
Questo metodo è stato scelto in combinazione all'architettura basata su rete modulare anche per mitigare il Catastrophic Forgetting.
- **Data Pruning**: Metodologia di Machine Unlearning adottata per una rimozione selettiva dei moduli obsoleti o inquinati da attacchi adversarial.

## Installazione e Utilizzo

**1. Prerequisiti**

È necessaria l'installazione di *Argus* per una fase iniziale di preprocessing dei dati e le dipendenze Python:

```bash
pip install -r requirements.txt
```

Oltre alle librerie standard, il progetto utilizza [Tesseract][https://github.com/s2labres/tesseract-ml-release], una libreria specializzata per il testing realistico di modelli di sicurezza attraverso lo split temporale dei dati.

**2. Preprocessing**

Il framework utilizza il dataset [MCFP (Malware Capture Facility Project)](https://www.stratosphereips.org/datasets-malware). Da li è possibile scaricare i pacchetti di rete (PCAP) usando *data_extractor.py* per alcuni di essi.

Dopodiché, nella cartella contenente i PCAP si procede con l'esecuzione di *Argus*:

```bash
for file in <cartella>/*.pcap; do argus -F argus.conf -r "$file" -w "${file%.pcap}.argus"; done

for file in <cartella>/*.argus; do ra -F ra.conf -r "$file" > "${file%.argus}.csv"; done
```

**3. Esecuzione del Test**
Il file ```main.py``` esegue tre scenari principali: **Concept Drift** (senza CL), **Continual Learning** (senza MU) e l'approccio proposto con **Machine Unlearning**.

```bash
python main.py
```

Per visualizzare risultati e osservazioni, leggere la [Tesi](./tesi_lm.pdf) riguardante il progetto.

## Riferimenti

* **Libreria Tesseract**: [s2labres/tesseract-ml-release](https://github.com/s2labres/tesseract-ml-release) - Utilizzata per l'analisi temporale e la mitigazione del bias.
* **Dataset MCFP**: [Stratosphere IPS](https://www.stratosphereips.org/datasets-malware) - Malware Capture Facility Project.
* **Tesi Magistrale**: [qui](./tesi_lm.pdf)