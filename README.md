# CECN Label Studio
## Labeling data with Label Studio

This will import the JSON-formatted list of data points in each file in the input path. The files should look like this:
```json
[
  {
    "tweet_text": "Opossum is great",
    "ref_id": "<tweet_id>",
    "meta_info": {
      "timestamp": "2020-03-09 18:15:28.212882",
      "location": "North Pole"
    }
  }
]
```
set up and run projects in Label Studio