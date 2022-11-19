Shift Happens task: ImageNet-M
==============================================

This task evaluates ImageNet-M, a 68-example evaluation split of the multi-label evaluation set
composed of "major" mistakes that several top-performing models make; we believe this subset is
one that future image classification models should achieve near perfect accuracy on, and provides
three clear benefits: (1) we attempt to comprehensively-label all examples for multi-label annotations
to prevent the need to review novel correct predictions, (2) we endeavor to maintain and provide a
way for the public to add new correct predictions; (3) the evaluation set is small enough to encourage
completeness and allow the community to inspect their own errors.