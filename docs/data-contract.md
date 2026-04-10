# Data Contract

## NormalizedMarketEvent

V2'de `fields` dict birincil veri kaynağıdır. Builder `event.fields.get(field_name)`
üzerinden ham değerlere erişir.

```python
NormalizedMarketEvent(
    event_time=...,
    exchange="binance",
    symbol="BTCUSDT",
    stream_type="mark_price",
    value=100.0,        # LEGACY — yeni builder kullanmaz
    fields={            # V2 birincil kaynak
        "mark_price": 100.0,
        "event_delta": 0.01,
        "index_price_if_available": 99.95,
    },
    ingest_metadata={"source": "..."},
)
```

`value` alanı legacy compat için tutulmaktadır. Yeni kod `fields` dict'ini kullanmalıdır.

## DatasetSpec

`stream_universe` tüm stream ailelerinin birleşimidir.
`available_streams_by_exchange` exchange'in yapısal olarak sunduğu stream'leri listeler.

### V2 Eklemeleri

`field_catalogs: list[StreamFieldCatalog]`
: Her stream ailesi için zorunlu ham field isimlerinin listesi.
  Boş bırakılırsa `REQUIRED_FIELDS_BY_STREAM` sabitinden doldurulur.

`availability_by_contract: dict[str, dict[str, bool]]`
: `exchange → stream → bool` düzeyinde fine-grain override.
  `available_streams_by_exchange`'e göre daha ayrıntılı kontrol sağlar.

Örnek:

```yaml
availability_by_contract:
  binance:
    open_interest: false   # binance OI akışı yok — yapısal eksiklik
  bybit:
    bbo: false             # bybit BBO yok
```

### Erişilebilirlik Önceliği

1. `availability_by_contract[exchange][stream]` varsa o değer geçerlidir
2. Yoksa `available_streams_by_exchange[exchange]` listesine göre hesaplanır

## Exchange-Stream-Field Availability Semantics

| Durum | `unavailable_by_contract` | `padding` | `missing` | `stale` |
|---|---|---|---|---|
| Tarih dışı bucket | False | **True** | False | False |
| Yapısal erişilemez | **True** | False | False | False |
| Erişilebilir — event yok | False | False | **True** | False |
| Erişilebilir — stale event | False | False | False | **True** |
| Geçerli değer | False | False | False | False |

Bu masklar birbirini dışlar; öncelik sırası yukarıdan aşağıyadır.

## Stream-Specific Field Families

Zorunlu field setleri:

```python
REQUIRED_FIELDS_BY_STREAM = {
    "trade":         ["price", "qty", "side_or_signed_flow_proxy", "event_delta", "count_or_burst"],
    "bbo":           ["bid_price", "ask_price", "bid_size", "ask_size", "spread", "mid", "imbalance_inputs"],
    "mark_price":    ["mark_price", "event_delta", "index_price_if_available"],
    "funding":       ["funding_rate", "next_funding_time", "funding_update_age"],
    "open_interest": ["open_interest", "oi_delta", "oi_update_age"],
}
```

Bir event'te eksik field `NaN` olarak işaretlenir; builder `missing=True` set eder.

## Exchange Stream Coverage (Fixture / Default Profile)

Fixture config `available_streams_by_exchange` tanımına göre:

| Exchange | bbo | trade | mark_price | funding | open_interest |
|---|---|---|---|---|---|
| binance | yes | yes | yes | yes | no |
| bybit | no | no | yes | no | yes |
| okx | no | no | yes | no | no |

`no` = `available_streams_by_exchange` veya `availability_by_contract` ile yapısal olarak dışlanmış.
Tensor'da bu koordinatlar `unavailable_by_contract=True` alır.

## S3 Compact Storage Layout

```text
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/meta.json
exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/quality_day.json
```

State dosyası: `compacted/_state.json`
