Task description copied from https://objectnet.dev/index.html.

Task Description
================
Ready to help develop the next generation of object recognition algorithms that have
robustness, bias, and safety in mind. Controls can remove bias from other datasets 
machine learning, not just vision.

ObjectNet is a large real-world test set for object recognition with control 
where object backgrounds, rotations, and imaging viewpoints are random.

Most scientific experiments have controls, confounds which are removed from the data, 
to ensure that subjects cannot perform a task by exploiting trivial correlations 
in the data. Historically, large machine learning and computer vision datasets 
have lacked such controls. This has resulted in models that must be fine-tuned for 
new datasets and perform better on datasets than in real-world applications. When 
tested on ObjectNet, object detectors show a 40-45% drop in performance, with 
respect to their performance on other benchmarks, due to the controls for biases. 
Controls make ObjectNet robust to fine-tuning showing only small performance increases.

Dataset Creation
=================
We develop a highly automated platform that enables gathering datasets with 
controls by crowdsourcing image capturing and annotation.

Evaluation Metrics
===================
Robust accuracy: correct classification of the images from new viewpoints on new backgrounds.

Expected Insights/Relevance
============================
The accuracy of pretrained ImageNet models decreases significantly on the proposed dataset.

Access
======
You can access zip file with data via website. The zip file has a password to ensure
that everyone is aware of our unusual license. The password is: objectnetisatestset.

Data
====
The dataset is hosted on https://objectnet.dev/download.html with a backup 
at https://www.dropbox.com/s/raw/cxeztdtm16nzvuw/objectnet-1.0.zip.

License
=======
Plese read this section, ObjectNet has an unusual license!

ObjectNet is free to use for both research and commercial applications.
The authors own the source images and allow their use under a license derived
from Creative Commons Attribution 4.0 with only two additional clauses.

1. ObjectNet may never be used to tune the parameters of any model.
2. Any individual images from ObjectNet may only be posted to the web including their 1 pixel red border.

---

1. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models.
    Barbu, Andrei and Mayo, David and Alverio, Julian and Luo, William and Wang, Christopher and Gutfreund, Dan and Tenenbaum, Josh and Katz, Boris. 2019.