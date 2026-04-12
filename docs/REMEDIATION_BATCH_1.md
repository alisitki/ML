# Remediation Batch 1 Report

**Date:** 2026-04-12  
**Scope:** Verification-first remediation of audit findings before any new feature work

## 1. Verification summary

### Concrete findings

| ID | Durum | Kanıt | Etki | Şimdi mi sonra mı |
|---|---|---|---|---|
| F01 | doğrulandı | `src/quantlab_ml/training/trainer.py` NumPy kullanıyor; repo içinde aktif `torch` trainer yok; `torch` yalnız `ml` extras içinde | governance drift; stack kararı muğlak kalıyordu | şimdi: karar kaydı eklendi, migration sonra |
| F02 | kısmen doğrulandı | yerel `.env` dosyası mevcut ve secret görünümlü değerler içeriyor; `.env` ignore ediliyor; `git ls-files` ve `git rev-list --all -- .env` mevcut repoda path history göstermiyor | local secret hygiene riski; mevcut repo görünümünde doğrulanmış git-history leak yok | şimdi: hygiene guardrail, dış aksiyon ayrı |
| F03 | doğrulandı | `TrajectoryBuilder` derived surface üretiyordu; aktif feature extractor bunu kullanmıyordu | observation-path bilgi kaybı; Batch 1 öncesi model derived sinyalleri görmüyordu | şimdi: düzeltildi |
| F04 | doğrulandı | trainer örnekleri yalnız `bundle.splits["train"]` ve `["validation"]` üzerinden kuruyor; `split_artifact.folds` okunmuyor | walk-forward cross-validation gap; overfitting riski açık kalıyor | sonra |
| F05 | kısmen doğrulandı | fixture config yalnız `1m×4`; QL-004 somut gap listesi yoktu; runtime enforcement tam değil | observation schema enforcement eksik ama auditteki tüm iddialar aynı ağırlıkta değildi | şimdi: gap listesi yazıldı, sıkılaştırma sonra |
| F06 | kısmen doğrulandı | `selection/service.py` minimal; fakat mevcut repo aşaması için daha zengin selector zorunlu değil | sınırlı selector kabiliyeti | sonra |
| F07 | kısmen doğrulandı | legacy scalar reward alanları ilk venue’dan türetiliyor; aktif reward uygulaması venue-aware context kullanıyor | compat yüzeyinde drift, aktif reward path’te değil | sonra |
| F08 | kısmen doğrulandı | build-time policy state önceki `selected_venue` üzerinden taşınıyor; evaluation sırasında gerçek state güncelleniyor | build-time state ile replay-time state arasında açıklık ihtiyacı | sonra |
| F09 | doğrulandı | repo genelinde logging yoktu | sessiz bozulma riski, gözlemlenebilirlik zayıftı | şimdi: düzeltildi |
| F10 | doğrulandı | compat adapter applicable reward yoksa `-0.001` sentinel döndürüyor; aktif production path bunu kullanmıyor | legacy compat ceza drift’i | sonra |
| F11 | yanlış pozitif | `builder.py` içindeki `break`, en iç stream döngüsünden çıkar; Python semantiği audit yorumunu desteklemiyor | gerçek bug kanıtı yok | şimdi: kapatıldı |
| F12 | kısmen doğrulandı | `EvaluationEngine.evaluate(..., split=...)` override kabul ediyor; CLI default olarak validation çalıştırıyor ve final-test flag’i yok | API seviyesinde potansiyel footgun, varsayılan akışta değil | sonra |
| F13 | doğrulandı | `_recompute_index()` tüm kayıtları tekrar okuyor | registry büyüdüğünde performans borcu | sonra |
| F14 | doğrulandı | `configs/training/` altında production preset config yok | QL-004 kapanışı eksik kalıyor | sonra |

### Drift notes

| ID | Durum | Not |
|---|---|---|
| D-1 | doğrulandı | F01 ile aynı drift; Batch 1A’da karar kaydı eklendi |
| D-2 | doğrulandı | F04 ile aynı drift; fold consumption hâlâ açık gap |
| D-3 | doğrulandı | F03 ile aynı drift; Batch 1B’de kapatıldı |
| D-4 | doğrulandı | `MomentumBaselineTrainer` sessiz alias’tı; Batch 1A’da deprecated shim’e çevrildi |
| D-5 | kısmen doğrulandı | compat katmanı yaşamaya devam ediyor; aktif training path değil |
| D-6 | doğrulandı | `PROJECT_STATE.md` ve `BACKLOG.md` QL-004’ü muğlak bırakıyordu; Batch 1A’da netleştirildi |

## 2. Confirmed findings

- Aktif trainer gerçekten NumPy tabanlı; bu durum artık `docs/DECISIONS.md` içinde açık geçici drift olarak kayıtlı.
- Derived surface üretimi ile feature extraction arasında gerçek kopukluk vardı; shared extractor şimdi derived channel değerlerini kullanıyor.
- Walk-forward fold’lar kalıcı artifact içinde var ama active trainer fold-aware development iteration yapmıyor.
- Repo gerçekten logging’sizdi; builder, trainer ve registry/store artık temel structured log üretiyor.
- `MomentumBaselineTrainer` ismi sessiz alias nedeniyle yanıltıcıydı; artık deprecated shim olarak işaretli.
- Production observation preset config eksikliği ve runtime compatibility tightening ihtiyacı QL-004 altında açık backlog gap’leri olarak yazıldı.

## 3. False positive / partial findings

- Secret exposure bulgusu repo-history leak olarak doğrulanmadı; mevcut repo görünümünde `.env` tracked değil ve `.env` path’i için history kanıtı yok. Buna rağmen yerel secret dosyası gerçek görünümlü değerler taşıdığı için dış aksiyon gerektirir.
- Derived-surface `break` bug iddiası yanlış pozitif çıktı; mevcut `break` yalnız stream döngüsünü sonlandırıyor.
- `EvaluationEngine` final untouched test’e override ile çağrılabilir, ancak mevcut CLI akışı bunu varsayılan veya kolay bir kullanıcı yüzeyi olarak açmıyor.
- Minimal selector, legacy reward snapshot scalar’ları ve compat adapter ceza değeri gerçek ama aktif repo aşamasında sınırlı etkili driftler olarak kaldı.

## 4. Uygulanan patch listesi

- `.env.example` eklendi ve `.gitignore` secret dosyalarını daha sıkı kapsayacak şekilde güncellendi.
- NumPy-vs-PyTorch mevcut durumu için `D-011` karar kaydı eklendi.
- `PROJECT_STATE.md` Batch 1 tamamlandı durumuna, `BACKLOG.md` ise QL-004 somut gap listesine göre güncellendi.
- `TrajectoryBuilder`, `LinearPolicyTrainer` ve `LocalRegistryStore` içine stdlib `logging` tabanlı hafif structured log iskeleti eklendi.
- `MomentumBaselineTrainer` sessiz alias olmaktan çıkarılıp deprecated shim’e çevrildi.
- `observation_feature_vector()` derived surface channel değerlerini deterministik sırayla feature vektörüne dahil edecek şekilde güncellendi.
- Logging, deprecated shim ve derived feature wiring için yeni testler eklendi.

## 5. Manuel dış aksiyonlar

- Eğer yerel `.env` içindeki credential’lar hâlâ aktifse rotate edilmeleri gerekir.
- Bu remediation sırasında git history rewrite yapılmadı; mevcut repo görünümünde `.env` path history leak kanıtı bulunmadı.
- Secret management tarafında vault/CI-injected env benzeri dış mekanizma gerekiyorsa bu repo dışı iş olarak planlanmalıdır.

## 6. Kalan riskler

- Runtime contract tightening Batch 2’ye ertelendi; runtime hâlâ full scale-spec veya derived-feature contract mismatch’lerini reddetmiyor.
- Walk-forward folds metadata olarak mevcut ama trainer fold-consumption yapmıyor; backtest-overfitting riski bu noktada açıkça kalıyor.
- Production observation preset config hâlâ eksik.
- Aktif trainer hâlâ NumPy tabanlı; PyTorch hedefi korunuyor ama migration yapılmadı.

## 7. Sonraki küçük batch önerisi

- Batch 2’yi yalnız observation/runtime compatibility tightening olarak tut:
  - runtime contract tightening
  - compatibility tags
  - runtime rejection rules
  - mismatch testleri
- Sonraki ayrı küçük batch’te walk-forward fold consumption’ın gerçek development CV gereksinimi olup olmadığı karar altına alınmalı.
