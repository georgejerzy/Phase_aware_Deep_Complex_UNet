gcloud compute scp  --recurse jerzybalcerzak@"tensorflow-2-3-20210529-232035":/home/jerzybalcerzak/Phase_aware_Deep_Complex_UNet/model_save/* --project "filler-words-rm"  --zone "europe-west1-b" model_save/
gcloud compute scp  --recurse jerzybalcerzak@"tensorflow-2-3-20210529-232035":/home/jerzybalcerzak/Phase_aware_Deep_Complex_UNet/training_logs/* --project "filler-words-rm"  --zone "europe-west1-b" training_logs/
