# QuantLab Prod Path Hidden Blocker Audit

## A) Yönetici Özeti

- **En büyük gizli risk:** Verinin `TrajectoryRecord` (Pydantic/JSON) ve raw float kopyaları arasında sürekli dönüşüme girmesidir. QL-021 sadece \`train\` sırasındaki darboğazı çözecektir, ancak \`evaluate\` ve \`validation\` fazlarında Pydantic nesneleri ve on-the-fly \`runtime_bridge.decide()\` çağrıları sebebiyle aynı single-threaded GIL darboğazı farklı bir formatta tekrar karşımıza çıkacaktır.
- **QL-021 öncesi neden önemli:** QL-021'i tasarlarken (binary tensor formatına geçerken) sadece `train` fazını düşünürsek, evaluation ve validation fazları için JSONL parse ve on-the-fly çıkartım mekanizması kalır. Bu da run süresinin train kısmını hızlandırırken, validation kısmında tekrar saatlerce süren bir tıkanmaya neden olur.
- **Yeniden hangi duvara çarpabilir:** QL-021 sonrası GPU training 10 dakikaya inebilir ancak \`evaluate\` adımı CPU-bound olarak 60 dakika sürebilir. Ayrıca, checkpoint mekanizmasının tamamen eksik olması, instance veya spot kesintilerinde (örneğin OOM veya SSH kopması) 6-10 saatlik (build + stream) işin tamamen çöpe gitmesi anlamına gelebilir.

---

## B) Hidden Blocker Listesi

### 1. `EvaluationEngine` CPU-bound GIL darboğazı (Yüksek Risk)
- **Ne zaman patlar:** QL-021 sonrası model training hızlandığında, validation ve evaluate adımları eski hızında (8-10 saniye/batch) kalarak toplam run'ı yine uzatacaktır.
- **Kanıt:** `engine.py` içinde `evaluate_records` metodunda her step için `decision = self.runtime_bridge.decide(artifact, step.observation)` çağrısı yapılıyor. Bu, `observation_feature_vector()` (680K liste operasyonu) çağrısını iç içe tekrar kullanır.
- **Neden önemli:** `train` hızlansa bile her epoch sonunda yapılan `evaluate` tüm `validation_split`'ini single-thread Python üzerinden geçirir.
- **Minimum doğru çözüm:** QL-021 tensor cache yapısını `validation` ve `evaluate` sırasında da okunabilecek şekle getirmek ve model inference'ını Python loop'undan toplu tensor pass (batch) işlemine dönüştürmek.
- **Şimdi mi sonra mı:** QL-021 ile birlikte (Validation/Evaluate tensor batch desteği de dahil edilmeli).

### 2. S3 / JSONL Read Amplification ve Garbage Collection Gecikmesi (Orta Risk)
- **Ne zaman patlar:** Daha geniş veri seti (çoklu gün, bütün semboller) ile remote instance'da bellek darboğazı başladığında.
- **Kanıt:** `streaming_store.py` içindeki `iter_records` fonksiyonunda Pydantic nesnesi yaratılıp siliniyor ve `if yielded % _GC_EVERY_N_RECORDS == 0: gc.collect()` çağrısı var. Bu, büyük allocation/de-allocation sebebiyle CPU zamanının `gc`'de kaybedilmesine sebep oluyor.
- **Neden önemli:** JSON objelerinden numpy yaratmak Python heap'inde devasa bir dalgalanma yaratır.
- **Minimum doğru çözüm:** Float dizilerini (ve metadataları) saf binary olarak okumak (`np.load(mmap_mode="r")` veya `torch.load()`), JSON parse'ı sadece manifest için bırakmak.
- **Şimdi mi sonra mı:** QL-021 kapsamında `JSONL` formatının tamamen `train`/`evaluate` path'inden çıkarılması ile.

### 3. Checkpoint ve Resume Eksikliği (Yüksek Risk)
- **Ne zaman patlar:** GPU üzerinde bir hafıza hatası veya spot instance kapanması durumunda.
- **Kanıt:** `trainer.py` içinde `_train_streaming_epoch` epoch'lar boyunca uzun süren döngüler içeriyor, ancak dışa kaydedilen bir epoch-level checkpoint mekanizması yok (`best_parameters` sadece hafızada tutulup en son diske dump ediliyor).
- **Neden önemli:** Uzun süren remote run'larda veri kaybı = zaman/maliyet kaybı.
- **Minimum doğru çözüm:** `trainer.py`'ye opsiyonel epoch bazlı state dict kaydetme / yükleme (resume_from_checkpoint) parametresi eklemek.
- **Şimdi mi sonra mı:** Sonraki adım. QL-021 sonrası daha büyük run'lara geçilmeden hemen önce yapılmalı.

---

## C) Hot Path Risk Tablosu

| Path | Risk Sınıfı | Neden Darboğaz / Ölçekleme Riski |
|---|---|---|
| **Build** | Bellek dalgalanması (`_Index` vs `TrajectoryRecord`) | `builder.py` max_episode_steps adımını tuttuktan sonra diske dump eder. Ancak binlerce paralel sembol ve stream için index, RAM'i doldurabilir (OOM riski). |
| **Train** | Single-threaded GIL (Feature Materialization) | Python list `extend()` ve `np.asarray()` sebebiyle GPU boşta bekliyor. QL-021 bunu gidermek üzerine tasarlandı. |
| **Validation** | On-the-fly step inference | Validasyon da aynı event-by-event Pydantic modeli ve inference'ı (runtime bridge) kullandığı için `train` hızlansa bile zaman kaybedilecek en büyük ikinci nokta. |
| **Evaluate** | On-the-fly step inference | Aynı şekilde, model eval path de step-by-step ilerliyor. Numpy batching yok. |
| **Serialization** | JSONL Pydantic Object Overhead | Her feature'u base64 listeler olarak metne dönüştürüp `\n` ile saklamak disk ve decode/CPU amplification. |
| **Artifact Collection** | Memory-to-disk kopukluğu | `best_parameters` tamamen bittikten sonra diske yazılıyor. Hata olursa policy uçar. |
| **Remote Rerun** | Spot toleransı yok | 8 saatlik run, 7. saatte patlarsa her şey 0'dan başlıyor. |

---

## D) Prod-vs-Compat Drift Tablosu

| Path/Component | Rolü | Yanlış Kullanım Riski |
|---|---|---|
| `TrajectoryDirectoryStore` | **PROD** | - |
| `TrajectoryStore` | **COMPAT / TEST-ONLY** | Hata ile prod datası verildiğinde OOM sebebi. CLI'da `--trajectories` argümanının bir dosya (JSON) yerine, `TrajectoryDirectoryStore.is_trajectory_directory` ile düzgün yakalanmaması riski (Şu an CLI'da kontrol var ancak auto-detect riskli). |
| `LocalFixtureSource` | **COMPAT / TEST-ONLY** | Yanlış ortam tanımı ile test datası okuyarak canlı pipeline tetiklemesi. |
| `S3CompactedSource` | **PROD** | `load_events` stream olsa da alttaki obje indirme sırasında belleği yutması (pyarrow / gzip kullanımı). |
| `compat_matrix_first` | **COMPAT / LEGACY** | `LinearPolicyTrainer` içinde `if isinstance(bundle, TrajectoryBundle)` kontrolü ile in-memory veri akışı yoluna sızabiliyor. Çok büyük verilerde belleği kilitleyebilir. |

---

## E) QL-021 İçin Pre-refactor Guardrail Listesi

QL-021 uygulanırken kaçınılması gereken tasarım hataları:
- **Kaçınılması Gereken 1:** Sadece Train split için binary yaratıp Validation'ı JSONL bırakmak. (Validation performansı batar).
- **Kaçınılması Gereken 2:** Binary formatı seçerken, her `TrajectoryRecord` için veya her `step` için diske binlerce ufak `.pt` dosyası yazmak. (Disk I/O ve Inode darboğazı başlar). Büyük sharded bloklar yazılmalı (Örn: batch_X_001.pt).
- **Kaçınılması Gereken 3:** Tensorları hala CPU'da parçalı okuyup, Python list üzerinden GPU'ya koymak.
- **Zorunlu Acceptance:** QL-021 uygulandığında `builder.build_to_directory()` hem JSONL hem Pre-computed binary tensor diske koymalı (eski legacy ve evaluate kodlarını bozmamak için geçici olarak ya da kalıcı metadata storage olarak).
- **Görünür Metrik:** QL-021 sonrasındaki ilk run'da `nvidia_train.csv` üzerinde GPU kullanımının %90+ seviyesine oturduğunu (avg util > %70) `vmstat us` değerinin ise GPU bekleme seviyelerine inmesi gösterilmeli. Sadece "batch/sec" artmamalı, GPU saturation kanıtlanmalı.

---

## F) Son Karar

"QL-021’e girmeden önce repo için en önemli görünmeyen risk **aynı GIL/Python feature extraction darboğazının validation ve evaluate fazlarında on-the-fly (adım adım) model inference yapılırken tekrarlanmasıdır**. Bunu önlemenin en doğru yolu **QL-021 tensör ön-hesaplama (pre-compute) adımının hem observation_schema'yı uygulayan train verisine hem de validation/evaluate verisine uygulanması ve `EvaluationEngine`'in batch-tensor inference yapacak şekilde refactor edilmesidir**."
