# anime_vfr

[English documentation](README_ENG.md)

`anime_vfr` converte sorgenti MKV ibride, tipicamente DVD anime NTSC, in video progressivo mantenendo il trattamento corretto per ogni parte del contenuto. Una stessa sorgente può contenere film telecinato, materiale progressivo e movimento realmente interlacciato: applicare un unico IVTC o un bob globale non è sempre corretto.

La pipeline produce un output VFR usando tre strategie operative:

- `match_keep_pts`: ricostruisce i frame con TFM e conserva tutti i timestamp sorgente;
- `match_decimate`: usa TFM e rimuove soltanto la ridondanza confermata dal mapping TDecimate;
- `bob_expand`: ricostruisce i singoli field per il materiale realmente interlacciato.

Le etichette 24/30/60 fps nel report descrivono la cadenza osservata; non decidono da sole quale strategia applicare.

## Requisiti e configurazione

Il progetto è configurato per macOS/Homebrew. Prima dell'uso controlla [config.py](config.py), in particolare:

- i path di `mkvmerge`, `mkvextract`, `vspipe`, Python, MediaInfo, FFmpeg e FFprobe;
- `ENCODER_BIN` e `ENCODER_PARAMS`.

La configurazione inclusa usa `ffmpeg-full`, `libx265`, preset `fast` e CRF 20. L'encoder deve accettare Y4M da standard input e produrre un file video compatibile con Matroska.

Servono inoltre:

- Python con `numpy` e accesso allo stesso ambiente VapourSynth usato da VSPipe;
- moduli Python `vsdeinterlace`, `vsaa`, `vstools` e `vskernels` con le relative dipendenze;
- plugin VapourSynth BestSource, TIVTC, fmtconv, Sneedif/NNEDI3 e quelli richiesti da `QTempGaussMC`;
- MKVToolNix, MediaInfo e FFmpeg.

La traccia video sorgente deve avere track ID Matroska `0`: è la traccia dalla quale la pipeline estrae i timestamp.

Verifica minima dell'ambiente:

```bash
/opt/homebrew/bin/python3.14 -c "import vapoursynth as vs; import numpy; print(vs.__version__)"
/opt/homebrew/bin/vspipe --version
/opt/homebrew/bin/mkvmerge --version
```

## Uso rapido

Prima di codificare una sorgente nuova è consigliata un'analisi completa:

```bash
python3.14 anime_vfr.py "/path/episodio.mkv" --analyze-only --keep-work
python3.14 anime_vfr.py "/path/episodio.mkv" --output "/path/encoded"
```

È possibile passare anche una cartella; vengono elaborati gli MKV presenti direttamente al suo interno.

```bash
python3.14 anime_vfr.py "/path/serie" --output "/path/encoded"
```

## Opzioni

```text
anime_vfr.py source [opzioni]
```

| Opzione | Comportamento |
| --- | --- |
| `source` | File MKV o cartella contenente MKV. |
| `--analyze-only` | Esegue analisi, classificazione, dedup richiesto, timecode e VPY, senza encode o mux. |
| `--report` | Misura un MKV già prodotto; deve essere usato senza altre opzioni. |
| `--output PATH` | Cartella degli MKV finali. |
| `--work-dir PATH` | Radice per le cartelle di lavoro delle singole run. |
| `--keep-work` | Conserva gli artefatti intermedi. |
| `--bob [NUM/DEN]` | Forza bob CFR su tutto il titolo; default `60000/1001`. |
| `--bob-chapters LIST` | Forza bob sui capitoli 1-based indicati, per esempio `4,5`. |
| `--bob-range LIST` | Forza bob sui range temporali indicati, per esempio `22:30-23:50`. |
| `--field-order tff\|bff` | Ordine dei field per TFM/QTGMC; default `tff`. |
| `--progressive-dedup [N]` | Deduplica una sorgente già progressiva senza TIVTC o classificazione. |
| `--dedup [N]` | Abilita il dedup opzionale sui segmenti matched/decimated. |
| `--resize WIDTHxHEIGHT` | Imposta la risoluzione esatta di output. |
| `--yuv444` | Produce YUV 4:4:4 10 bit anziché YUV 4:2:0 10 bit. |
| `--threads N` | Imposta i thread VapourSynth e prefetch; default `os.cpu_count()`. |
| `--analysis-workers N` | Imposta i worker delle metriche NumPy; default uguale a `--threads`. |
| `--additional-vpy PATH` | Appende uno snippet VPY al clip finale. |
| `--frames N` | Codifica i primi `N` frame output. |
| `--frames A-B` | Codifica il range output half-open `[A,B)`. |
| `--strip-audio` | Esclude le tracce audio dal mux. |
| `--strip-sub` | Esclude le tracce sottotitoli dal mux. |

### Analisi e report

`--analyze-only` esegue la stessa decisione usata dall'encode. Il report contiene:

- una tabella delle strategie ponderata sul numero di frame sorgente;
- una tabella delle stesse strategie ponderata sulla durata PTS;
- tutti gli intervalli decisionali con frame sorgente/output, tempi, strategia, cadenza osservata e motivazioni disponibili;
- statistiche dei drop strutturali e del dedup opzionale.

Con `--keep-work` restano disponibili anche i file diagnostici per frame e per run, il VPY finale e i timecode. Senza `--keep-work` la cartella della run viene eliminata dopo la stampa del report.

```bash
python3.14 anime_vfr.py "/path/episodio.mkv" --analyze-only --keep-work
```

`--report` non esegue il classificatore. Estrae i timestamp dal track ID Matroska `0` di un MKV esistente e riassume finestre temporali come 24, 30, 60 fps, `other`, `VFR` o `unknown`. Stampa sia la distribuzione per numero di frame output sia quella per durata. È una misura del risultato muxato, non una spiegazione delle decisioni prese dalla pipeline.

```bash
python3.14 anime_vfr.py "/path/encoded/episodio.mkv" --report
python3.14 anime_vfr.py "/path/encoded" --report
```

### Override bob

`--bob` salta TFM, TDecimate, classificatore e dedup, ricostruisce tutto il titolo con bob e usa un mux CFR. Il rapporto opzionale deve essere `NUM/DEN`; per esempio, `--bob 50/1` per una sorgente 25i da portare a 50p.

`--bob-chapters` e `--bob-range` intervengono invece solo sulle parti indicate e lasciano il resto alla classificazione automatica. I range accettano secondi, `MM:SS` o `HH:MM:SS` e sono separati da virgole. Gli override locali sono incompatibili con `--bob`, `--progressive-dedup` e `--report`.

```bash
python3.14 anime_vfr.py "/path/episodio.mkv" --bob-chapters 4
python3.14 anime_vfr.py "/path/episodio.mkv" --bob-range 22:30-23:50,10:00-10:20
```

### Dedup

`--dedup [N]` compatta hold o duplicati visivi sui rami matched/decimated dopo la scelta della strategia primaria. Non sostituisce i drop strutturali di `match_decimate`. `N` limita il numero massimo di frame consecutivi compattabili in un singolo output e vale 2 se omesso.

`--progressive-dedup [N]` applica soltanto dedup e timing a una sorgente già progressiva. Non deve essere usato su materiale ancora interlacciato o telecinato.

### Resize, formato e filtri aggiuntivi

Senza `--resize` vengono mantenute le dimensioni codificate e il display aspect ratio sorgente. Con `--resize`, la risoluzione richiesta viene trattata come pixel quadrati: la pipeline non calcola automaticamente una risoluzione dal SAR.

`--additional-vpy` viene eseguito dopo assemblaggio, resize e conversione nel formato richiesto. Lo snippet riceve `clip`, deve riassegnarlo e non deve cambiare numero o ordine dei frame, perché i timecode sono già stati calcolati. Non deve chiamare `set_output()`.

Esempio minimo:

```python
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
```

### Selezione dei frame

`--frames` seleziona frame della timeline output, quindi dopo classificazione, decimazione ed eventuale dedup. La pipeline analizza comunque l'intera sorgente prima di applicare il range. Il limite finale è esclusivo.

Audio e sottotitoli vengono ritagliati in stream copy usando gli stessi limiti temporali del video. In questa modalità capitoli e allegati Matroska non vengono conservati; ciò include gli eventuali font allegati richiesti dai sottotitoli ASS.

## Timing e sicurezza delle transizioni

La pipeline separa matchability, ridondanza e cadenza osservata. TDecimate viene usato soltanto nei run matchable per i quali esiste un mapping di ridondanza validato; negli altri run ricostruibili vengono conservati tutti i PTS.

Le transizioni tra matched e bob vengono chiuse usando le dipendenze dei field indicate dai match TFM, evitando che un frame ricostruito usi un field appartenente al lato bob della frontiera.

I timecode finali derivano dalla timeline sorgente. Il file V2 contiene un timestamp per ogni frame output più un timestamp terminale, necessario per conservare esattamente anche la durata dell'ultimo frame dopo bob, decimazione, dedup o `--frames`. Una durata non quantizzabile può essere mantenuta nei percorsi matched, ma impedisce il bob sicuro di quel frame.

## Output e file di lavoro

Se `--output` non è specificato, un file `episodio.mkv` produce `episodio_1.mkv` accanto alla sorgente. Con `--output`, viene mantenuto il nome originale nella cartella scelta; la pipeline rifiuta di sovrascrivere direttamente il sorgente.

Ogni elaborazione usa una sottocartella univoca dentro `--work-dir` o, per default, dentro `work` nella cartella di output. Questo evita collisioni tra run parallele. In un mux completo vengono riutilizzati dal sorgente audio, sottotitoli, capitoli e allegati, salvo le esclusioni richieste.

## Struttura del progetto

```text
anime_vfr.py   entrypoint CLI
pipeline.py    analisi, classificazione e orchestrazione
branches.py    costruzione dei rami TFM/TDecimate
segments.py    mapping e segmentazione operativa
dedup.py       dedup sui rami selezionati
timecodes.py   validazione PTS e timecode finali
report.py      report pre-encode e post-encode
media.py       metadata e timestamp sorgente
encode.py      encode e mux
contracts.py   contratti e validazione runtime
config.py      binari e parametri configurabili
```

## Licenza

GNU General Public License v3.0. Vedi [LICENSE](LICENSE).

## Buy me a coffee!

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/simonefil)
