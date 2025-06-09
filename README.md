# Üretim Planlama Modelleri Python Projesi Kurulum ve Kullanım Kılavuzu

### Canlı demo: https://yzupky.streamlit.app/ 

## Kurulum 

1. **Python Kurulumu**
   - Python 3.8 veya üzeri bir sürüm kurulu olmalıdır.
   - [Python İndir](https://www.python.org/downloads/)

2. **Gerekli Paketlerin Kurulumu**
   - Terminal veya komut satırında proje klasörüne gelin:
     ```bash
     cd /home/alper/PycharmProjects/upky
     ```
   - Tüm bağımlılıkları yüklemek için:
     ```bash
     pip install -r requirements.txt
     ```

## Modellerin Çalıştırılması

Her model ayrı bir Python dosyasıdır. Örneğin Model 1'i çalıştırmak için:

```bash
python model1_mixed_planning.py
```

Diğer modeller için de benzer şekilde:
- Model 2: `python model2_overtime_production.py`
- Model 3: `python model3_batch_production.py`
- Model 4: `python model4_dynamic_programming.py`
- Model 5: `python model5_outsourcing_comparison.py`
- Model 6: `python model6_seasonal_planning.py`

## Streamlit Arayüzü ile Kullanım

Proje, tüm modelleri tek bir arayüzde çalıştırmak için bir Streamlit uygulaması içerir.

### Streamlit ile Başlatma

1. Terminalde proje klasörüne gelin:
   ```bash
   cd /home/alper/PycharmProjects/upky
   ```
2. Streamlit uygulamasını başlatın:
   ```bash
   source .venv/bin/activate
   streamlit run streamlit_app.py
   ```
3. Tarayıcıda açılan arayüzden istediğiniz modeli seçip parametreleri girerek çalıştırabilirsiniz.

- Sonuçlar tablo ve grafik olarak arayüzde görüntülenir.
- Her modelin parametreleri arayüzde kolayca değiştirilebilir.

## Sonuçlar
- Her model çalıştırıldığında, terminalde tablo halinde sonuçlar ve toplam maliyet görüntülenir.
- Ayrıca, modelin türüne göre otomatik olarak grafiksel çıktı (üretim, stok, işçi sayısı vb.) açılır.

## Parametre Değişikliği
- Her modelin başında parametreler Python kodunda tanımlıdır.
- Ana parametre dosyası olan `parametreler.yaml` üzerinden temel ayarlar değiştirilebilir.
- Streamlit arayüzünden de parametreler etkileşimli olarak değiştirilebilir.

## Notlar
- Grafiklerin açılması için `matplotlib` yüklü olmalıdır (requirements.txt ile otomatik yüklenir).
- Tablo çıktısı için `tabulate` kütüphanesi kullanılır.
- Optimizasyon modelleri için `pulp` kütüphanesi gereklidir.

## Model Özellikleri ve Güncellemeler

### Model 3: Toplu Üretim ve Stoklama
- Fiziksel stok seviyeleri artık hiçbir zaman 0'ın altına düşmez
- İşe alım maliyetleri toplam maliyete dahil edilmiştir
- Grafikler stok ve karşılanmayan talebi doğru gösterir

### Model 4: Dinamik Programlama
- Talebe göre optimum işçi sayısını belirler
- İlk dönemde sadece ihtiyaç duyulan kadar işçiyi işe alır
- İşçi değişim kısıtlarını doğru uygular

### Model 6: Mevsimsellik ve Dalga
- Sıfır işçiyle başlar ve talebi karşılamak için ihtiyaç duyulan işçileri işe alır
- İşe alım maliyetleri toplam maliyete dahil edilmiştir

## Karşılaştırma Tablosu (Streamlit)

- Karşılaştırma tablosunda tüm modeller aynı parametreler ile çalıştırılır ve sonuçlar karşılaştırılabilir.
- Her modelin toplam maliyeti, işçilik maliyeti, üretim maliyeti, stok maliyeti, karşılanmayan talep maliyeti ve işe alım/çıkarma maliyetlerini içerir.
- Grafiksel karşılaştırmalar maliyet bileşenlerinin analizini kolaylaştırır.

## Hangi Model Hangi Senaryoda Kullanılır?

- **Model 1: Karma Planlama**
  - Talep ve işgücü dalgalanmalarının yüksek olduğu, hem iç üretim hem fason esnekliğinin gerektiği durumlar için uygundur.
- **Model 2: Fazla Mesaili Üretim**
  - İşgücü sabit, talep dalgalı ve kısa vadeli artışlar fazla mesaiyle karşılanabiliyorsa tercih edilir.
- **Model 3: Toplu Üretim ve Stoklama**
  - Talebin öngörülebilir ve üretim kapasitesinin sabit olduğu, stok tutmanın sorun olmadığı durumlar için uygundur.
- **Model 4: Dinamik Programlama**
  - İşgücü planlamasının ve işçi değişim maliyetlerinin önemli olduğu, uzun vadeli ve değişken talep yapısında kullanılır.
- **Model 5: Dış Kaynak Kullanımı Karşılaştırması**
  - Farklı tedarikçi seçeneklerinin ve kapasite kısıtlarının olduğu, dış kaynak tercihinde maliyet karşılaştırması yapılacaksa uygundur.
- **Model 6: Mevsimsellik ve Talep Dalgaları**
  - Talebin mevsimsel dalgalandığı, stok ve üretim optimizasyonunun önemli olduğu, esnek işgücü planlaması gereken sektörlerde kullanılır.

---

Herhangi bir hata veya eksiklikte, terminaldeki hata mesajını kontrol ederek eksik paketi yükleyebilir veya parametreleri gözden geçirebilirsiniz.

