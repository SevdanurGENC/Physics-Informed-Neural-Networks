# Physics-Informed-Neural-Networks
PINNs - Physics Informed Neural Networks
 
## 🧠 PINNs Nedir? — Sıfırdan İleri Seviyeye Tam Rehber

---

## 📌 BÖLÜM 1: Temel Kavramlar (Hiç bilmeyenler için)

### 1.1 Sinir Ağı (Neural Network) Nedir?

Sinir ağı, bir **fonksiyon tahmincisidir**. Yani:

```
Girdi → [Sinir Ağı] → Çıktı
```

Örneğin: Bir evin metrekaresini verirsen, fiyatını tahmin eder.

Sinir ağı bunu **parametreler (ağırlıklar)** öğrenerek yapar. Bu öğrenme işlemine **eğitim** denir.

---

### 1.2 Kayıp Fonksiyonu (Loss Function) Nedir?

Sinir ağı ne kadar yanılıyor? Bunu ölçen şey **kayıp fonksiyonu**dur.

```
Gerçek değer: 100
Tahmin:        90
Kayıp:         (100 - 90)² = 100
```

Amaç: Kaybı **minimize etmek** (sıfıra yaklaştırmak).

---

### 1.3 Klasik Sinir Ağının Problemi

Klasik sinir ağı sadece **veriden öğrenir**. Yani:

- Çok veri lazım 📊
- Fizik kurallarını bilmez 🚫
- Saçma sonuçlar üretebilir (enerjinin korunumu ihlali gibi)

---

## 📌 BÖLÜM 2: Diferansiyel Denklemler (PINNs'in kalbi)

### 2.1 Diferansiyel Denklem (PDE/ODE) Nedir?

Fizik olaylarını matematiksel olarak tanımlayan denklemlerdir.

**Örnek — Isı Denklemi:**
```
∂u/∂t = α · ∂²u/∂x²
```

Bu denklem şunu söyler:
> "Bir çubuktaki sıcaklık, zamana göre nasıl değişir?"

**PDE = Partial Differential Equation (Kısmi Diferansiyel Denklem)**
- Birden fazla değişkene bağlı (zaman + uzay gibi)

**ODE = Ordinary Differential Equation (Adi Diferansiyel Denklem)**
- Sadece bir değişkene bağlı

---

### 2.2 Neden Zor?

Çoğu gerçek hayat denklemi **analitik (kapalı form) çözüme** sahip değil.
Yani el ile çözemiyorsun. Bilgisayar ile sayısal yöntemler kullanılıyor (FEM, FDM gibi) — ama bunlar da çok **hesaplama kaynağı** istiyor.

---

## 📌 BÖLÜM 3: PINNs — Temel Fikir

### 3.1 Büyük Fikir 💡

> **"Fizik bilgisini sinir ağının kayıp fonksiyonuna ekle!"**

Klasik kayıp:
```
Loss = Veri Hatası
```

PINN kaybı:
```
Loss = Veri Hatası + Fizik Hatası
```

Fizik hatası nedir?
> Sinir ağının tahmini, fizik denklemini ne kadar ihlal ediyor?

---

### 3.2 Somut Örnek — Burgers Denklemi

```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
```

Bu denklem sıvı akışını tanımlar.

PINNs şunu yapar:
1. Sinir ağı `u(x, t)` fonksiyonunu tahmin eder
2. Bu tahminin türevlerini hesaplar (otomatik diferansiyasyon ile!)
3. Denklemi kontrol eder: sol taraf - sağ taraf = **fizik hatası (residual)**
4. Bu hatayı da **kayıp fonksiyonuna ekler**

---

### 3.3 PINNs Mimarisi

```
        [x, t]
           │
    ┌──────▼──────┐
    │  Sinir Ağı  │  ← Tam bağlantılı katmanlar
    │  (MLP)      │
    └──────┬──────┘
           │
          u(x,t)   ← Tahmin
           │
    ┌──────▼──────────────────────┐
    │  Otomatik Diferansiyasyon   │
    │  ∂u/∂t, ∂u/∂x, ∂²u/∂x²   │
    └──────┬──────────────────────┘
           │
    ┌──────▼──────────────────────┐
    │  KAYIP FONKSİYONU           │
    │  L = L_data + L_physics     │
    └─────────────────────────────┘
```

---

## 📌 BÖLÜM 4: Otomatik Diferansiyasyon

Bu PINNs'in **sihirli silahı**!

### 4.1 Nedir?

Sinir ağları zaten backpropagation için türev hesaplar. PINNs bunu **girdi değişkenlerine göre** de kullanır.

PyTorch'ta örnek:
```python
import torch

x = torch.tensor([2.0], requires_grad=True)
t = torch.tensor([1.0], requires_grad=True)

# Sinir ağı tahmini
u = model(torch.cat([x, t]))

# Otomatik türev
u_t = torch.autograd.grad(u, t, create_graph=True)[0]
u_x = torch.autograd.grad(u, x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]

# Fizik hatası (örn. ısı denklemi: u_t - α*u_xx = 0)
physics_residual = u_t - alpha * u_xx
```

---

## 📌 BÖLÜM 5: Kayıp Fonksiyonu Detayı

```
L_total = λ₁·L_data + λ₂·L_BC + λ₃·L_PDE
```

| Terim | Açıklama |
|-------|----------|
| `L_data` | Ölçüm noktalarındaki veri hatası |
| `L_BC` | Sınır koşulları hatası (Boundary Conditions) |
| `L_PDE` | Fizik denkleminin ihlali (residual) |
| `λ` | Her terimin ağırlığı |

---

## 📌 BÖLÜM 6: PINNs'in Avantajları ve Dezavantajları

### ✅ Avantajlar

- **Az veri ile çalışır** — Fizik bilgisi eksiği kapatır
- **Sınır koşulları doğal gelir** — Kayıp fonksiyonuna eklenir
- **Ters problemleri çözebilir** — Bilinmeyen parametreleri öğrenebilir
- **Mesh (ızgara) gerekmez** — Klasik sayısal yöntemlerin aksine

### ❌ Dezavantajlar

- **Yavaş eğitim** — Her adımda türev hesabı pahalı
- **Hiperparametre hassasiyeti** — λ değerleri kritik
- **Karmaşık geometrilerde zorlanır**
- **Başarısızlık modları var** (causality problemi gibi)

---

## 📌 BÖLÜM 7: Ters Problem (Inverse Problem)

Bu PINNs'in en güçlü özelliklerinden biri!

### Klasik (İleri) Problem:
```
Denklem + Başlangıç koşulları → Çözüm u(x,t)
```

### Ters Problem:
```
Ölçüm verisi → Bilinmeyen fizik parametresi (ν, α, ...)
```

**Örnek:** Su akışı gözlemliyorsun ama viskoziteyi (ν) bilmiyorsun.
PINNs hem `u(x,t)`'yi hem de `ν`'yü aynı anda öğrenir!

```python
# ν eğitilebilir parametre olarak tanımla
nu = torch.nn.Parameter(torch.tensor([0.01]))
```

---

## 📌 BÖLÜM 8: Tam Kod Örneği (1D Isı Denklemi)

```python
import torch
import torch.nn as nn
import numpy as np

# ─── Model ───────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
alpha = 0.01  # Isı difüzyon katsayısı

# ─── Eğitim Döngüsü ──────────────────────────────────
for epoch in range(10000):
    optimizer.zero_grad()
    
    # 1) Veri kaybı (başlangıç koşulu: u(x,0) = sin(πx))
    x_data = torch.rand(100, 1)
    t_data = torch.zeros(100, 1)
    u_pred = model(x_data, t_data)
    u_true = torch.sin(np.pi * x_data)
    loss_data = torch.mean((u_pred - u_true)**2)
    
    # 2) Fizik kaybı (∂u/∂t = α·∂²u/∂x²)
    x_phys = torch.rand(1000, 1, requires_grad=True)
    t_phys = torch.rand(1000, 1, requires_grad=True)
    u = model(x_phys, t_phys)
    
    u_t = torch.autograd.grad(
        u, t_phys, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]
    u_x = torch.autograd.grad(
        u, x_phys, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x, x_phys, grad_outputs=torch.ones_like(u_x),
        create_graph=True)[0]
    
    residual = u_t - alpha * u_xx
    loss_phys = torch.mean(residual**2)
    
    # 3) Toplam kayıp
    loss = loss_data + loss_phys
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```

---

## 📌 BÖLÜM 9: İleri Seviye Konular

### 9.1 Failure Modes (Başarısızlık Modları)

**Causality Problemi:**
> PINNs zaman boyutunu doğru sırayla öğrenmeyebilir. t=0'dan t=T'ye kadar nedensellik zinciri kopmabilir.

**Çözüm:** Casual PINNs — zamanı parçalara böl, sırayla eğit.

---

### 9.2 Pareto Cephesi (Çoklu Hedef Optimizasyonu)

```
L = λ₁·L_data + λ₂·L_PDE
```

λ değerlerini değiştirince farklı denge noktaları çıkar:
- Yüksek λ₁ → Veriye daha sadık
- Yüksek λ₂ → Fiziğe daha sadık

Bu **Pareto cephesini** keşfetmek önemlidir.

---

### 9.3 Fractional PINNs

Standart türevler yerine **kesirli türevler** kullanan genişleme:
```
∂^α u / ∂t^α = ...   (0 < α < 1)
```
Uzun hafızalı sistemler (viskoelastik malzemeler, anomal difüzyon) için.

---

### 9.4 Delta PINNs

Tekil (singular) çözümler için:
> Dalga kırınması, şok dalgaları gibi süreksizlik içeren problemler.

---

## 🗺️ Öğrenme Yol Haritası

```
Seviye 1  →  Sinir ağı temelleri (MLP, backprop, kayıp)
Seviye 2  →  Diferansiyel denklemler (ODE/PDE temelleri)
Seviye 3  →  Otomatik diferansiyasyon (PyTorch autograd)
Seviye 4  →  Basit PINN implementasyonu (1D ODE)
Seviye 5  →  2D PDE çözümü (Burgers, Navier-Stokes)
Seviye 6  →  Ters problem çözümü
Seviye 7  →  İleri seviye: Failure modes, Pareto, Fractional PINNs
```

---

## 📚 Kaynaklar

| Kaynak | Açıklama |
|--------|----------|
| [Raissi et al. 2019](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) | Orijinal PINN makalesi |
| [DeepXDE kütüphanesi](https://deepxde.readthedocs.io) | Hazır PINN framework'ü |
| [Steve Brunton YouTube](https://www.youtube.com/@Eigensteve) | Video serileri |

--- 
