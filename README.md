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
- Her modelin başında parametreler Python kodunda tanımlıdır. Kendi senaryonuza göre bu değerleri değiştirebilirsiniz.

## Notlar
- Grafiklerin açılması için `matplotlib` yüklü olmalıdır (requirements.txt ile otomatik yüklenir).
- Tablo çıktısı için `tabulate` kütüphanesi kullanılır.
- Optimizasyon modelleri için `pulp` kütüphanesi gereklidir.

## Kaynaklar
- Detaylı model açıklamaları ve parametreler için `upky Blueprint.md` dosyasına bakınız.

## Karşılaştırma Tablosu (Streamlit)

- Karşılaştırma tablosunda kullanıcıdan **yalnızca 6 parametre** alınır:
  - `demand` (Aylık talep)
  - `working_days` (Aylık çalışma günü)
  - `holding_cost` (Stok maliyeti)
  - `outsourcing_cost` (Fason maliyeti)
  - `labor_per_unit` (Birim işgücü)
  - `stockout_cost` (Karşılanmayan talep maliyeti)
- Diğer tüm parametreler modellerde sabit olarak tanımlıdır.
- Karşılaştırma tablosu ve grafikler, bu 6 parametreye göre otomatik güncellenir.

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
  - Talebin mevsimsel dalgalandığı, stok ve üretim optimizasyonunun önemli olduğu sektörlerde kullanılır.

---

Herhangi bir hata veya eksiklikte, terminaldeki hata mesajını kontrol ederek eksik paketi yükleyebilir veya parametreleri gözden geçirebilirsiniz.
