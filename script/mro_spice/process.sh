# source activate radar_env

# python /home/wbl/code/MARS/Mars_new/script/mro_spice/hirise2mk_online.py \
#   --pid ESP_069731_2055 --out /home/wbl/code/MARS/Mars_new/script/mro_spice/result/ESP_069731_2055.tm \
#   --download --dest /media/wbl/Elements/paper_experiments/Mars/kernals/MRO/

# python /home/wbl/code/MARS/Mars_new/script/mro_spice/hirise2mk_online.py \
#   --pid ESP_075559_2055 --out /home/wbl/code/MARS/Mars_new/script/mro_spice/result/ESP_075559_2055.tm \
#   --download --dest /media/wbl/Elements/paper_experiments/Mars/kernals/MRO/



# python /home/wbl/code/MARS/Mars_new/script/mro_spice/mro_pose_timeseries.py   \
#   --tm /home/wbl/code/MARS/Mars_new/script/mro_spice/result/ESP_069731_2055.tm   \
#   --start "2021-06-10T00:00:00"  --end   "2021-06-10T23:59:59"   --step 60   \
#   --sc-frame MRO_SPACECRAFT   \
#   --out /home/wbl/code/MARS/Mars_new/script/mro_spice/result/mro_pose_20210608_27_60s.csv


# python /home/wbl/code/MARS/Mars_new/script/mro_spice/mro_pose_timeseries.py   \
#   --tm /home/wbl/code/MARS/Mars_new/script/mro_spice/test.tm   \
#   --start "2006-11-22T00:00:00"   --end   "2006-11-22T23:59:59"   --step 60   \
#   --sc-frame MRO_SPACECRAFT   \
#   --out /home/wbl/code/MARS/Mars_new/script/mro_spice/result/mro_pose_20210608_27_60s.csv


# python /home/wbl/code/MARS/Mars_new/script/mro_spice/extract_hirise_from_img.py \
#    /media/wbl/ZX2_WBL/data/Mars/祝融号着陆区/ESP_069731_2055/ESP_069731_2055_RED0_0.IMG

# seqname="ESP_069731_2055"
# mkdir -p /media/wbl/Elements/paper_experiments/Mars/new/eo_setup/$seqname/
# python /home/wbl/code/MARS/Mars_new/script/mro_spice/generate_hirise_inputfile.py \
#   /media/wbl/ZX2_WBL/data/Mars/祝融号着陆区 \
#   $seqname \
#   /media/wbl/Elements/paper_experiments/Mars/new/eo_setup/$seqname.txt \
  # "/media/wbl/Elements/paper_experiments/Mars/new/eo_setup/$seqname/"

seqnames=(
  "ESP_069731_2055" "ESP_075559_2055" "ESP_075625_2055" "ESP_077511_2055"
)
for seqname in ${seqnames[@]};do
  source activate radar_env

  # mkdir -p /media/wbl/Elements/paper_experiments/Mars/new/eo_setup/$seqname/

  # python /home/wbl/code/MARS/Mars_new/script/mro_spice/generate_hirise_inputfile.py \
  #   /media/wbl/ZX2_WBL/data/Mars/祝融号着陆区 \
  #   $seqname \
  #   /media/wbl/Elements/paper_experiments/Mars/new/eo_setup/$seqname.txt \
  #   "/media/wbl/Elements/paper_experiments/Mars/new/eo_setup/$seqname/"

  python /home/wbl/code/MARS/Mars_new/script/mro_spice/hirise2mk_online.py \
    --pid $seqname --out /media/wbl/Elements/paper_experiments/Mars/new/eo_setup/${seqname}_pz.txt \
    --download --dest /media/wbl/Elements/paper_experiments/Mars/kernals/MRO/
done



