# CECN Label Studio
## Labeling data with Label Studio

I have made a backup of all annotations so don't worry if you accidentally remake the containers, just try not to
touch docker-compose up or anything that will restart the containers, because it will wipe out the annotations.

To run annotator agreement analysis:
```shell
docker attach label-studio-dev
python ls/annotator_agreement.py pv
```
it will print the head of the annotations common between all annotators, the shape of the dataframe (how many data points
were in common in the rows), the nominal alpha, and how to interpret it.