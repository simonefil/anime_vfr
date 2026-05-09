# anime_vfr

`anime_vfr` nasce per un problema tipico di molti DVD anime NTSC: dentro lo stesso episodio possono convivere sezioni film telecinate recuperabili a 23.976 fps, sezioni realmente interlacciate a 59.94 campi/s, crediti o titoli digitali sovrapposti, CG, pan generati in video, cambi di cadenza e hard telecine non uniforme. In questi casi il file e' spesso tutto 480i/29.97 a livello di contenitore o MPEG-2, ma il contenuto reale non e' tutto della stessa natura.

Il caso semplice e' un DVD soft-telecined o hard-telecined regolare: si applica IVTC e si torna a 23.976 fps. Il caso opposto e' una sorgente interlacciata pura: si applica bob deinterlace e si conserva il movimento a 59.94 fps. `anime_vfr` serve per la zona intermedia, cioe' per sorgenti ibride in cui entrambe le decisioni sono corrette solo localmente. E' una casistica frequente nei DVD anime NTSC di serie TV lunghe e prodotte in transizione analogico/digitale: nel lavoro su questo script e' stata verificata su `Bleach`, mentre discussioni tecniche pubbliche riportano problemi analoghi o vicini su DVD di `Naruto`, `Naruto Shippuden` e `One Piece`. Guide storiche citano inoltre DVD ibridi come `X TV`, `Azumanga Daioh` e `Super GALS`, con opening/ending, sezioni digitali o parti pure interlacciate mescolate a materiale telecinato.

Le soluzioni globali hanno compromessi pesanti. Un IVTC applicato a tutto l'episodio preserva bene il materiale film, ma sulle sezioni realmente interlacciate produce judder, drop errati o combing residuo. Un bob globale risolve il movimento 60i, ma trasforma anche il film in 60p interpolato, aumentando i frame e peggiorando la pulizia dei dettagli senza recuperare la struttura originale. Una conversione CFR unica a 23.976 o 29.97 costringe sempre una parte del materiale a un timing sbagliato: o si perde fluidita' nelle parti video, o si duplicano/interpolano frame nel film.

La strategia di `anime_vfr` e' diversa: classificare il contenuto per frame/segmento, costruire un output progressivo VFR e usare il trattamento corretto per ogni zona. Le parti film passano da IVTC/decimazione; le parti interlacciate reali passano da bob deinterlace; il mux finale usa timecode Matroska per mantenere il timing originale senza forzare tutto a un solo framerate costante.

## Installazione

Prima di eseguire lo script, apri `config.py` e adatta i path dei binari alla tua macchina:

- `MKVMERGE`, `MKVEXTRACT`, `VSPIPE`, `PYTHON_BIN`, `MEDIAINFO`, `FFMPEG`
- `ENCODER_BIN`
- `ENCODER_PARAMS`

I parametri di compressione inclusi sono solo un preset operativo di esempio. Devi verificarli e modificarli in base all'encoder scelto, alla GPU/CPU disponibile e al livello qualitativo desiderato. In particolare, se non usi `NVEncC`, devi cambiare sia `ENCODER_BIN` sia `ENCODER_PARAMS` con una riga compatibile con il tuo encoder.

Assicurati che `python` sia il Python dell'ambiente VapourSynth, cioe' quello in grado di importare `vapoursynth`. Servono: Python packages `numpy`, `vsdeinterlace`, `vsaa`, `vstools`, `vskernels` e relative dipendenze; plugin VapourSynth BestSource, TIVTC, Vinverse, Sneedif/NNEDI3 OpenCL e i plugin richiesti da `QTempGaussMC` come MVTools/RGTools/RemoveGrain o equivalenti della propria distribuzione; binari esterni `VSPipe`, `mkvmerge`, `mkvextract`, `MediaInfo`, `ffmpeg`; encoder video compatibile con input Y4M da pipe, ad esempio `ffmpeg`, Rigaya `NVEncC` per NVIDIA NVENC, Rigaya `QSVEncC` per Intel Quick Sync, Rigaya `VCEEncC` per AMD VCE/VCN/AMF, oppure Rigaya `rkmppenc` per Rockchip MPP. I path e i parametri dell'encoder si configurano in `config.py`.

Verifiche minime:

```powershell
python -m pip install numpy
python -c "import vapoursynth as vs; import numpy; print(vs.__version__)"
python -c "import vapoursynth as vs; c=vs.core; print(hasattr(c,'bs'), hasattr(c,'tivtc'), hasattr(c,'vinverse'), hasattr(c,'sneedif'))"
python -c "from vsdeinterlace.qtgmc import QTempGaussMC; from vsaa import NNEDI3; print(NNEDI3(opencl=True)._deinterlacer_function)"
```

Nel codice il branch bob usa:

```python
QTempGaussMC(
    clip,
    basic_bobber=NNEDI3(nsize=4, nns=4, qual=2, opencl=True),
    tff=True,
    basic_tr=3,
    final_tr=2,
    source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED,
).deinterlace()
```

Con `vsaa.NNEDI3(opencl=True)` il wrapper seleziona il backend OpenCL `core.lazy.sneedif.NNEDI3`; con `opencl=False` userebbe invece il backend CPU `core.lazy.znedi3.nnedi3`.

## Uso Rapido

```powershell
cd C:\Users\Simone\anime_vfr
python anime_vfr.py "C:\video\episodio.mkv" --analyze-only
python anime_vfr.py "C:\video\episodio.mkv" --output "D:\encoded"
```

## Parametri

Sintassi:

```text
anime_vfr.py source [opzioni]
```

| Parametro               | Descrizione breve                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `source`                | File MKV sorgente, oppure cartella di MKV.                                                  |
| `--report`              | Analizza MKV gia' prodotti leggendo i timestamps_v2; non esegue la pipeline.               |
| `--analyze-only`        | Esegue analisi, classificazione, dedup, timecode e VPY, ma salta encode/mux.               |
| `--bob`                 | Forza tutto il titolo a 60p bob e salta TIVTC/classificatore/dedup.                        |
| `--bob-chapters LIST`   | Forza a 60p bob uno o piu' capitoli, es. `4` o `4,5,6`.                                    |
| `--bob-range LIST`      | Forza a 60p bob uno o piu' range temporali, es. `22:30-23:50`.                             |
| `--progressive-dedup [N]` | Deduplica una sorgente gia' progressiva; `N` opzionale limita il run massimo.             |
| `--dedup [N]`           | Abilita il dedup sui segmenti film; `N` opzionale limita il run massimo.                   |
| `--output PATH`         | Cartella di output; se specificata mantiene il nome del file sorgente.                     |
| `--work-dir PATH`       | Cartella di lavoro; se omessa usa `<output>\work`.                                         |
| `--keep-work`           | Conserva i file intermedi invece di pulire la work dir a fine elaborazione.                |
| `--additional-vpy PATH` | Appende uno snippet VPY al pass finale; non e' un VPY standalone.                          |
| `--frames RANGE`        | Elabora solo un range di frame output, formato `N` oppure `A-B`.                           |
| `--strip-audio`         | Non muxa le tracce audio sorgenti.                                                         |
| `--strip-sub`           | Non muxa i sottotitoli sorgenti.                                                           |

### `source`

In modalita' normale e' un singolo MKV o una cartella di MKV da processare. Con `--report` puo' essere un MKV gia' prodotto oppure una cartella contenente MKV gia' prodotti.

```powershell
python anime_vfr.py "C:\video\episodio.mkv"
python anime_vfr.py "D:\encoded" --report
```

### `--report`

Non usa il classificatore e non ricostruisce la pipeline. Estrae i timestamps_v2 dal MKV finale, misura gli intervalli tra frame e stampa una distribuzione posteriore in classi `<24`, `24`, `24<x<60`, `60`, `>60`. Serve a controllare il risultato finale, non a decidere come encodare.

```powershell
python anime_vfr.py "D:\encoded\episodio.mkv" --report
python anime_vfr.py "D:\encoded" --report
```

### `--analyze-only`

Esegue la pipeline fino alla generazione dei timecode finali e dello script VPY, poi si ferma prima di encode e mux. Stampa su console classificazione 24/60, statistiche dedup, istogramma FPS pre-dedup su 20 bucket e istogramma drop dedup su 20 bucket.

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --analyze-only
python anime_vfr.py "C:\video\episodio.mkv" --analyze-only > analisi.txt
```

### `--bob`

Forza l'intero titolo a 60p bob. E' utile quando sai gia' che la sorgente e' interlacciata reale e non vuoi tentare recupero film. In questa modalita' non vengono eseguiti TFM/TDecimate, classificatore e dedup.

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --bob
```

### `--bob-chapters` e `--bob-range`

Forzano a 60p bob solo parti specifiche del titolo, lasciando il resto alla classificazione automatica. Servono quando una sezione e' nota a priori come movimento interlacciato reale, per esempio ending con credits o scroll verticali che devono rimanere fluidi. `--bob-chapters` usa indici capitolo 1-based letti dal MKV sorgente; `--bob-range` usa range temporali manuali nel formato `START-END`, separati da virgola.

Questi override vengono applicati dopo il classificatore e prima della segmentazione finale. Non usano la timeline decimata: i frame del range vengono forzati a `video_bob` sulla timeline sorgente, quindi ogni frame sorgente genera due frame output e i timecode dividono in due la durata reale del frame. Sono mutuamente esclusivi con `--bob`, `--progressive-dedup` e `--report`.

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --bob-chapters 4
python anime_vfr.py "C:\video\episodio.mkv" --bob-chapters 4,5,6
python anime_vfr.py "C:\video\episodio.mkv" --bob-range 22:30-23:50
python anime_vfr.py "C:\video\episodio.mkv" --bob-range 22:30-23:50,10:00-10:20
```

### `--progressive-dedup`

Usa solo la parte dedup/timecode su una sorgente gia' progressiva. Salta TIVTC, classificatore e bob: ogni frame sorgente viene trattato come frame progressivo valido, poi i frame visivamente duplicati vengono rimossi e la loro durata viene trasferita ai timecode VFR. Il valore opzionale `N` indica quanti frame consecutivi possono essere compattati al massimo; se scrivi solo `--progressive-dedup`, viene usato `N=2`.

Questa modalita' serve per sorgenti che non hanno bisogno di IVTC o deinterlace, ma contengono hold/duplicati reali che si vogliono compattare in VFR. Non e' adatta a sorgenti interlacciate o telecinate non gia' risolte.

```powershell
python anime_vfr.py "C:\video\progressivo.mkv" --progressive-dedup --analyze-only
python anime_vfr.py "C:\video\progressivo.mkv" --progressive-dedup 4 --analyze-only
python anime_vfr.py "C:\video\progressivo.mkv" --progressive-dedup --output "D:\encoded"
```

### `--dedup`

Abilita il dedup sui segmenti film della pipeline ibrida. Senza questo flag, la pipeline fa IVTC, classificazione 24/60, bob delle sezioni interlacciate reali e generazione dei timecode VFR, ma conserva tutti i frame film decimati. `--dedup` serve quando vuoi compattare hold e duplicati visivi trasferendone la durata ai timecode VFR. Il valore opzionale `N` indica quanti frame consecutivi possono essere compattati al massimo; se scrivi solo `--dedup`, viene usato `N=2`.

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --dedup
python anime_vfr.py "C:\video\episodio.mkv" --dedup 4
```

### `--output` e `--work-dir`

`--output` sceglie dove scrivere gli MKV finali. Se viene specificato, il file finale mantiene esattamente il nome del sorgente. Se viene omesso, l'output viene scritto nella stessa cartella del sorgente con suffisso `_1`, cosi' il file originale non viene sovrascritto.

`--work-dir` sceglie dove creare gli artefatti necessari alla lavorazione. Ogni processamento usa una sottocartella univoca, quindi piu' istanze possono condividere la stessa work dir senza sovrascrivere i file intermedi. Senza `--keep-work`, la sottocartella della run viene pulita al termine; gli output richiesti dall'utente non dipendono da `--keep-work`.

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --output "D:\encoded"
python anime_vfr.py "C:\video\episodio.mkv" --work-dir "E:\temp\anime_vfr"
```

Esempi di output:

```text
python anime_vfr.py "C:\video\episodio.mkv"
  -> C:\video\episodio_1.mkv

python anime_vfr.py "C:\video\episodio.mkv" --output "D:\encoded"
  -> D:\encoded\episodio.mkv
```

### `--additional-vpy`

Appende uno snippet VPY al pass finale generato da `anime_vfr`. Non e' un VPY autonomo: viene copiato dentro il pass2b dopo che la pipeline ha gia' creato il clip VFR finale.

Quando lo snippet viene eseguito, esistono gia':

- `import vapoursynth as vs`
- `core = vs.core`
- `clip`, cioe' il clip gia' assemblato dai segmenti 24p/60p, gia' ridimensionato a pixel quadrati e convertito in `YUV420P10`

Lo snippet lavora sulla variabile `clip` e la riassegna. Un caso tipico e' azzerare il flag interlacciato residuo e forzare il tag CFR tecnico prima dei filtri:

```python
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
clip = core.std.AssumeFPS(clip, fpsnum=30000, fpsden=1001)

from vstools import depth
clip = depth(clip, 16)

from vsdeband import placebo_deband
clip = placebo_deband(clip, radius=8.0, thr=3.0, iterations=4, grain=0.0)
```

Non ricaricare la sorgente con `VideoSource` e non chiamare `clip.set_output(0)`: `anime_vfr` lo aggiunge alla fine. Lo snippet deve mantenere numero e ordine dei frame, perche' i timecode VFR sono gia' stati generati.

Esempio di uso:

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --additional-vpy "C:\filtri\filtri.vpy"
```

Lo stesso modello e' quello usato dagli snippet generati da strumenti come `gto_crop_detect.py`: il file prodotto opera direttamente su `clip` e viene appeso al pass2b.

### `--frames`

Limita l'encode a un range di frame output. Con un singolo numero `N` processa `0-N`; con `A-B` processa l'intervallo indicato. Quando e' attivo, audio e sottotitoli vengono trimmati con FFmpeg prima del mux.

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --frames 1500
python anime_vfr.py "C:\video\episodio.mkv" --frames 100-5000
```

### `--strip-audio`, `--strip-sub`

`--strip-audio` e `--strip-sub` escludono rispettivamente audio e sottotitoli dal mux finale.

```powershell
python anime_vfr.py "C:\video\episodio.mkv" --strip-audio
python anime_vfr.py "C:\video\episodio.mkv" --strip-sub
```

## Funzionamento

La pipeline non decide il framerate finale guardando soltanto l'output di TDecimate. TIVTC viene usato per costruire una timeline film coerente e per estrarre segnali utili, ma la decisione 24p/60p viene presa sui frame sorgente con un classificatore multi-metrica. Questo e' necessario sulle sorgenti anime miste: scene quasi statiche, pattern incompleti o frame combed isolati possono ingannare un singolo indicatore. La pipeline incrocia piu' segnali e alla fine riduce tutto a una scelta binaria: film oppure bob.

Per prima cosa vengono letti metadati, SAR e timestamps_v2 della sorgente. I timestamps sorgente sono la base temporale: servono a generare i timecode finali e a chiudere correttamente anche l'ultimo segmento, evitando drift cumulativo rispetto all'audio. La risoluzione viene convertita a pixel quadrati mantenendo il display aspect ratio.

Il pass TIVTC iniziale produce le informazioni di match tra campi e decimazione. TFM cerca corrispondenze tra campi per recuperare i frame progressivi da materiale telecinato; TDecimate costruisce la timeline decimata. Durante il secondo pass viene creato un framemap che collega ogni frame della timeline decimata al frame sorgente da cui deriva. Questo collegamento e' essenziale: la classificazione lavora sulla sorgente, ma l'assemblaggio film lavora sulla timeline decimata.

Il classificatore considera ogni frame sorgente come centro di una finestra di dieci campi. In quella finestra misura quali campi sono quasi uguali al campo precedente tramite differenza luma 16x16: sotto soglia sono considerati corrispondenti. Il telecine 3:2 produce una struttura ciclica riconoscibile; per evitare falsi positivi il codice accetta come fase telecine solo il caso stretto con esattamente due match distanti cinque campi. Se questo pattern e' presente e il frame non e' combed, il frame viene classificato come 24p. Se invece e' combed, il flag TFM prevale e il frame viene trattato come 60i reale.

Il solo pattern non basta. La pipeline misura anche il movimento dei campi e un rapporto FFT sull'energia verticale a Nyquist del piano luma. L'energia a Nyquist intercetta l'alternanza tra righe, tipica dell'interlacciamento con movimento; valori molto bassi, insieme a movimento sufficiente, possono promuovere materiale progressivo/telecine anche quando il pattern ciclico non e' completo. Le scene molto statiche vengono trattate con prudenza: se quasi tutti i campi corrispondono e il movimento e' minimo, la decisione non viene presa come prova forte di 60i, perche' anche materiale progressivo fermo puo' avere campi quasi identici.

Dopo la classificazione iniziale vengono applicate passate di coerenza. Le ancore telecine isolate vengono scartate se non hanno abbastanza vicini coerenti; i piccoli cluster telecine circondati da 60i vengono riclassificati; i frame ambigui ereditano dalla densita' locale e dall'ancora piu' vicina, con un bias coerente con il motivo per cui erano ambigui. Infine, sui cluster 60i abbastanza lunghi viene fatta una verifica IVTC speculativa: si applica TFM lento su una subclip e si controlla quanti frame restano combed. Un cluster viene recuperato come telecine solo se IVTC lo pulisce e se il movimento medio e' sufficiente; questo evita di confondere scene statiche con vero recupero 24p.

Alla fine la pipeline normalizza in due classi operative. Solo i frame classificati `interlaced_60i` diventano `video_bob`; tutto il resto entra nel ramo `film`. Il framemap viene riscritto con questa scelta binaria e raggruppato in segmenti contigui. Nei segmenti bob vengono conservati gli indici sorgente esatti, perche' TDecimate puo' aver saltato frame sorgente dentro una stessa zona visiva e non bisogna reinserire frame sbagliati.

Gli override `--bob-chapters` e `--bob-range` intervengono in questo punto: non cambiano le metriche del classificatore, ma sostituiscono la decisione finale nei range scelti dall'utente. Sono pensati per casi editorialmente noti, come ending sempre da tenere a 60p, e vanno preferiti a euristiche globali quando la scelta dipende dalla struttura dell'episodio.

Il dedup lavora solo sui segmenti film. Ricostruisce lo stesso stream decimato che verra' usato nell'encode, confronta frame film consecutivi e raggruppa run di duplicati fino al limite configurato. Se trova, per esempio, una run 4-in-1, tiene un solo frame video ma allunga il timing tramite timecode. In questo modo il contenuto visivo non viene ripetuto inutilmente, ma la durata resta agganciata alla sorgente.

I timecode finali sono generati dalla timeline sorgente e dalla segmentazione. Nei segmenti film viene scritto un timestamp per ogni frame decimato tenuto; con dedup il timestamp successivo avanza della durata rappresentata dalla run. Nei segmenti bob ogni frame sorgente genera due frame output, quindi la durata sorgente viene divisa in due. L'ultimo timestamp viene chiuso usando la durata stimata dai timestamps sorgente, non un framerate teorico fisso.

Il VPY finale costruisce solo i rami necessari. Se ci sono segmenti film, crea il ramo TFM/TDecimate e applica Vinverse sui residui combed. Se ci sono segmenti 60p, crea il ramo QTempGaussMC con NNEDI3 OpenCL. I segmenti vengono poi assemblati con `core.std.Splice`. `AssumeFPS` nel VPY e' solo un tag tecnico per lo stream; il timing reale nel MKV finale e' determinato dai timecode Matroska muxati.

In modalita' normale `VSPipe` invia Y4M all'encoder configurato, poi `mkvmerge` muxa video, timecode VFR, audio e sottotitoli sorgenti. In `--analyze-only` la pipeline si ferma prima dell'encode ma produce comunque il report tecnico su standard output. In `--report`, invece, non viene rieseguita nessuna decisione: si misura soltanto il file finale gia' prodotto.

## Struttura del codice

```text
anime_vfr.py   entrypoint CLI
pipeline.py    orchestrazione, pass TIVTC, classificatore e VPY finale
config.py      path binari e parametri della pipeline
media.py       metadata video e timestamps sorgente
segments.py    framemap e segmentazione film/bob
dedup.py       dedup sui segmenti film
timecodes.py   generazione timecode VFR finali
report.py      report analyze-only e report posteriori
encode.py      encode e mux finale
utils.py       helper condivisi
```

## Note operative

- Il modello finale e' sempre binario: 24p film o 60p bob.
- `--analyze-only` e' il modo corretto per validare una sorgente nuova prima dell'encode.
- `--report` serve solo sui file gia' prodotti: misura i timestamps del MKV, non ripete la classificazione.
- Il dedup e' applicato solo ai segmenti film e modifica il numero di frame video, non la durata temporale.
- `--progressive-dedup` applica lo stesso principio a sorgenti gia' progressive, senza classificazione 24/60.
- Gli script `--additional-vpy` devono mantenere invariati numero e ordine dei frame.
- Se cambi encoder, aggiorna insieme `ENCODER_BIN` e `ENCODER_PARAMS` in `config.py`.

## Licenza

Questo progetto e' distribuito sotto licenza GNU General Public License v3.0. Vedi `LICENSE`.

## Buy me a coffee!

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/simonefil)
