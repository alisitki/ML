# Learning Surface

## V2 Tensor Düzeni

Gözlem yüzeyi düz `value_cube`'dan çok ölçekli ham tensor bloklarına evrilmiştir.
Her `RawScaleTensor` tam boyutu `(time, symbol, exchange, stream, field)` şeklindedir.

```text
raw_surface[scale_label] → RawScaleTensor
  shape: [num_buckets, n_symbols, n_exchanges, n_streams, n_fields]
  tensors: values | age | padding | unavailable_by_contract | missing | stale
```

Hiçbir exchange ortalaması veya stream'den tek scalar collapse yapılmaz.
Tüm ham field aileleri her koordinat için ayrı ayrı taşınır.

## Zaman Ölçekleri (Scale Preset)

Her scale kendi bucket, age ve mask mantığını bağımsız olarak korur.

| Profil       | Scale        | Bucket  |
|---|---|---|
| Smoke/Fixture | `1m`        | ×4      |
| Production   | `1m`         | ×8      |
| Production   | `5m`         | ×8      |
| Production   | `15m`        | ×8      |
| Production   | `60m`        | ×12     |

Production preset `configs/training/production.yaml`'da tanımlanacak.

## Field Katalogları

Her stream ailesinin zorunlu ham field seti `DatasetSpec.field_catalogs` içinde tanımlıdır.

| Stream          | Zorunlu Field'lar |
|---|---|
| `trade`         | `price, qty, side_or_signed_flow_proxy, event_delta, count_or_burst` |
| `bbo`           | `bid_price, ask_price, bid_size, ask_size, spread, mid, imbalance_inputs` |
| `mark_price`    | `mark_price, event_delta, index_price_if_available` |
| `funding`       | `funding_rate, next_funding_time, funding_update_age` |
| `open_interest` | `open_interest, oi_delta, oi_update_age` |

`NormalizedMarketEvent.fields` dict V2'de birincil veri kaynağıdır.
`value` alanı yalnızca legacy uyumluluk için tutulmaktadır.

## Mask Önceliği

Bir koordinat için mask değerleri birbirini dışlar:

```
padding → unavailable_by_contract → missing → stale → geçerli değer
```

- `padding`: yalnızca yetersiz tarihçe — `history_start`'tan önce olan bucket'lar
- `unavailable_by_contract`: yapısal olarak erişilemez `(exchange, stream)` çifti
- `missing`: erişilebilir ama bu adımda hiç event gelmedi
- `stale`: erişilebilir, event var, ama `stale_after_seconds`'ı geçmiş

## Derived Surface

Ham tensorların yerini almaz; `DerivedSurface` ayrı blok olarak eklenir.

V1 kapsam (target-centric):

| Kanal | Açıklama |
|---|---|
| `venue_pair_*_spread_A_B` | Target symbol için tüm exchange çiftleri arasında fiyat farkı (O(n²)) |
| `relative_move_T_vs_S`    | Target symbol vs configured universe'deki her diğer symbol |

## Action Feasibility

`action_mask: dict[str, bool]` → `ActionFeasibilitySurface`

Boyut: `action × venue × size_band × leverage_band → FeasibilityCell`

Yalnızca decision-time'da bilinen bilgiden üretilir. Future fiyat veya
sonraki reward timestamp kullanılmaz.

## Reward Context ve Timeline

Her step için venue-aware ekonomik durum:

- `RewardContext`: decision-time referans fiyatları, fee/slippage rejimleri,
  funding freshness, önceki pozisyon durumu
- `RewardTimeline`: `horizon_steps` adım için venue-specific referans serisi

Reward hesabı exchange-average yerine seçilen venue'nun verisi üzerinden yapılır.

## Policy State

Her step önceki step'ten inventory snapshot taşır:

```
previous_position_side | previous_venue | hold_age_steps | turnover_accumulator
```

Yalnızca geçmiş uygulanmış kararlardan ve decision-time-known state'ten türetilir.

## Compatibility Adapter

V1 davranışını emüle eden reduction fonksiyonları `contracts/compat.py`'de yaşar:

- `target_stream_series(obs, stream)` — exchange-ortalamalı V1 serisi
- `flat_value_cube(obs)` — ilk scale'in values listesi
- `snapshot_reference_price(snapshot)` — context veya legacy scalar

Bu modül çekirdek `contracts/__init__.py`'den dışa açılmaz; yalnızca
`training/compat_adapter.py` ve legacy test kodu import eder.

## Episode Length

V2 default: `max_episode_steps = 128`
V1 3-step chunking kaldırılmıştır.

## Evaluation Boundary

Training ve evaluation aynı reward kontratını paylaşır.

V2 replay'de:
- Fill assumption: `next_mark_price` (venue-specific timeline'dan)
- Action mask: `action_feasibility.to_flat_mask()`
- Reward: venue-specific context + timeline üzerinden
