# Notes on 12.02.2021

    - overleaf -> makale yazimi icin.
        - heuristic kalitesi
        - gercek veri kullanimi
        - farkimizi ortaya koymamiz lazim.
    - sonuclari tablo haline getirelim.
    - justify the parameters.
    - network kullanimini detaylandirmamiz lazim.
        - kullanicilarin aldiklari paketlern degisimi, ayni paketten ayni pakette 1 aylik periyotta gecis yapanlar arasindaki iliski, bu kisilere gonderilen mesaj ile ilgili mi?
        Paketler <- kullanici
                        |
                    kullanici
    - rolling horizon konusu bakalim.
        - burada bir onceki planlama ile yapilan gonderilem bir c-u-d matrisi alinir, haftalik kota hesaplanirken X + c-u-d toplamina bakmak lazim burada c-u-d matrisi window mantiginda olmali
    - 3 buyuk probleme deginmemiz lazim.

# Notes on 25.03.2021
from ezgi kurkcu to everyone:    2:28 PM
smsrep.dm_sms_mms_wappush
from ezgi kurkcu to everyone:    2:28 PM
smsrep.dm_ivr
from ezgi kurkcu to everyone:    2:28 PM
smsrep.dm_week
from ezgi kurkcu to everyone:    2:28 PM
rcdbde kampanya talep datası
from ezgi kurkcu to everyone:    2:29 PM
xddste mstrcommon.list_reports - uld bazlı abone datası
from ezgi kurkcu to everyone:    2:29 PM
subscriber id bazlı data
from ezgi kurkcu to everyone:    2:29 PM
bscs dumrul şemasındaki kota tabloları
from ezgi kurkcu to everyone:    2:30 PM
select * from DUMRUL.TCATQ_SEGMENT_DEF --kota da kontrol edilen segment lookup tablosu
select * from DUMRUL.TKODX  where TKOD like '%CATQ%' -- kota kategorileri  tanım  tablosu
edit  DUMRUL.TQUOTA_CHANNEL_DEF --kota kanal  tanım tablosu
select *from DUMRUL.TQUOTA_TOTAL_CATQ --toplama dahil olan kategoriler
select *   from dumrul.tcatqprf  -- kota limitlerinin turulduğu  tablosu
select  CCATQ as Kategori, tdesckod, CPAYTYPE as paymenttype, ctimeperiod as period,segmentid,CCATCHNL as channel    
from dumrul.tcatqprf t,   DUMRUL.TKODX  k where CCATQ =NVALUKOD  and TKOD like '%CATQ%' --- kanal, segment ve kategori bazlı limit tablosu
select * from  dumrul.tquota -- kota kullanımlarının insert edildiği tablo

from ezgi kurkcu to everyone:    2:30 PM
cdh şemasında optimizasyonun kendi tabloları var
from ezgi kurkcu to everyone:    2:31 PM
SMSREP.VW_DM_OPT_QUOTA_CAT_PRIORITY  - kota kategori ve tanıtım amacına göre öncelik
from ezgi kurkcu to everyone:    2:31 PM
SMSREP.VW_DM_OPT_QUOTA_CAT_PERCENT - kota kategori ve yüzdeleri
from ezgi kurkcu to everyone:    2:31 PM
smsrep.VW_OPT_MAX_SEND_COUNT -  kanal bazlı günlüx max gönderim count
from ezgi kurkcu to everyone:    2:39 PM
select * from SMSREP.VW_DM_IVR_OPT
from ezgi kurkcu to everyone:    2:39 PM
select * from SMSREP.VW_DM_SMS_MMS_WAPPUSH_OPT 



=============================
    - kullanicilarin aldiklari paketlern degisimi, ayni paketten ayni pakette 1 aylik periyotta gecis yapanlar arasindaki iliski, bu kisilere gonderilen mesaj ile ilgili mi?
        Paketler <- kullanici
                        |
                    kullanici
    - rolling horizon konusu bakalim.
        - burada bir onceki planlama ile yapilan gonderilem bir c-u-d matrisi alinir, haftalik kota hesaplanirken X + c-u-d toplamina bakmak lazim burada c-u-d matrisi window mantiginda olmali