# 24. AMP — Automatic Mixed Precision

Источник (ментор): [AMP examples — PyTorch 2.15](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)

В курсе Дубинина темы **нет**. Это последний «обязательный» кусок PyTorch-инфры перед своим треком (recsys/NLP): уметь читать и писать цикл обучения с AMP.

---

## 1. Зачем

Обычное обучение — почти всё в **`float32`**.

**AMP** = часть операций в **`float16` / `bfloat16`** (быстрее, меньше памяти на GPU), чувствительные места остаются в `float32`.

На практике AMP = два инструмента вместе:

1. **`torch.autocast`** — сам выбирает dtype для ops внутри блока  
2. **`torch.amp.GradScaler`** — масштабирует loss/градиенты, чтобы мелкие `float16`-градиенты не обнулялись (underflow)

> На **CPU** ускорение часто слабое; API тот же. На **CUDA** — основной смысл.  
> `GradScaler` критичен для **`float16`**. Для `bfloat16` часто можно без него (диапазон шире).

---

## 2. Типичный цикл (выучить наизусть)

```python
from torch.amp import autocast, GradScaler

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler(device.type)  # один раз на всё обучение

for epoch in epochs:
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x)
            loss = loss_fn(pred, y)

        # backward НЕ внутри autocast (так рекомендуют доки)
        scaler.scale(loss).backward()
        scaler.step(optimizer)   # unscale → step (или skip при inf/NaN)
        scaler.update()          # подстроить scale на следующий шаг
```

Шаги по смыслу:

| вызов | что делает |
|:---|:---|
| `autocast(...)` | forward + loss в смешанной точности |
| `scaler.scale(loss).backward()` | умножить loss на scale → градиенты тоже «крупнее» |
| `scaler.step(opt)` | разделить градиенты обратно; если inf/NaN — **пропустить** `optimizer.step` |
| `scaler.update()` | увеличить/уменьшить scale |

Веса модели по умолчанию остаются `float32`; меняется dtype **активаций/операций** внутри `autocast`.

---

## 3. Gradient clipping + AMP

Клипать нужно **уже unscaled** градиенты. Иначе порог `max_norm` будет «не тот».

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                       # один раз на optimizer за step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

Правило: **`unscale_` один раз** на optimizer между `backward` и `step`. Дважды → `RuntimeError`.

---

## 4. Gradient accumulation + AMP

Копим scaled-градиенты несколько микробатчей, `step`/`update` — только на полном effective batch:

```python
loss = loss / iters_to_accumulate
scaler.scale(loss).backward()

if (i + 1) % iters_to_accumulate == 0:
    # при желании: unscale_ + clip здесь
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

Пока копим — **не** `unscale_` и **не** `update` (иначе смешаются масштабы).

---

## 5. Что из доки знать «на уровне имени», не кодить сейчас

| раздел доки | тебе |
|:---|:---|
| Typical Mixed Precision Training | **обязательно** |
| Gradient clipping + `unscale_` | **обязательно** |
| Gradient accumulation | понять схему |
| Gradient penalty | позже (GAN и т.п.) |
| Multiple models / losses / optimizers | знать: `scale` на каждый loss, `step` на каждый opt, `update` **один** раз |
| Multiple GPUs / DataParallel | когда дойдёшь до multi-GPU |
| Custom autograd + `custom_fwd/bwd` | когда пишешь свои `Function` |

---

## 6. Чеклист «AMP закрыт»

- [ ] AMP = `autocast` + (обычно) `GradScaler`  
- [ ] forward/loss внутри `autocast`; `backward` снаружи через `scaler.scale(loss)`  
- [ ] порядок: `scale → backward → (unscale_ → clip) → step → update`  
- [ ] clipping только после `unscale_`  
- [ ] accumulation: `step`/`update` на границе effective batch  
- [ ] на CPU без CUDA цикл всё равно умеешь написать (API)  
