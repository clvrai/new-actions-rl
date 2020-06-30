rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p 22" data/vids  aszot@lim-b.usc.edu:~/nips2019_backup/data
rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p 22" data/logs  aszot@lim-b.usc.edu:~/nips2019_backup/data
rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p 22" data/trained_models  aszot@lim-b.usc.edu:~/nips2019_backup/data
