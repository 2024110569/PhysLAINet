## Overview

<p align="center">
  <b>PhysLAINet</b><br>
</p>
<p align="center" style="white-space: nowrap;">

  <img src="https://visitor-badge.laobi.icu/badge?page_id=2024110569.PhysLAINet" />

  <img src="https://img.shields.io/github/stars/2024110569/PhysLAINet?style=flat-square" />

  <img src="https://img.shields.io/github/forks/2024110569/PhysLAINet?style=flat-square" />

  <img src="https://img.shields.io/badge/status-under%20review-orange?style=flat-square" />

  <img src="https://img.shields.io/badge/license-academic%20use%20only-red?style=flat-square" />

  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" />

</p>

This repository contains supplementary materials for a submitted academic paper, including the self-designed models, comparative models, experimental datasets, and an end-to-end interactive web system built upon the method proposed in this paper. The data files are uploaded as a compressed package. The password is the email address of the first author. This ensures that only reviewers and editors can access the files during the review process. Once the manuscript is accepted, access will be available to all via the information provided in the paper. Furthermore, considering the extremely large size of raw image data (up to several hundred gigabytes), only structured data with pre-extracted features is uploaded here.

## Repository Contents
This repository provides the following resources for academic peer review (the list is non-exhaustive):
- The proposed self-designed models
- Mainstream comparative models for benchmark experiments
- Experimental datasets
- End-to-end web application developed based on the proposed method

---
## Note: Web system
### Backend Directory Structure
```text
backend/
  app/
    main.pyd
    processor.pyd
    schemas.pyd
  models/
    lai_model.onnx  # obtained through running physlainet.run()
    lai_model_scaler.joblib  # obtained through running physlainet.run()
  storage/
frontend/
  index.html
  styles.css
  app.js
```

### Start the Backend
Run the following command in PowerShell:
```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### Launch the Frontend
Open the file directly with a web browser:
```text
frontend/index.html
```
The frontend connects to `http://127.0.0.1:8000` by default.

---

## License & Usage Statement
All resources in this repository are **for academic peer review only** at present. Prior to the official publication of the corresponding paper, unauthorized reproduction, modification, commercial use, public distribution and secondary creation of any part of this project are strictly prohibited.

Once the paper is accepted and formally published, all resources will be fully open-sourced under the **MIT License**. Researchers may use, reference and modify the code and datasets for non-commercial academic research, provided that the original paper is properly cited.

The author retains all intellectual property rights of this project. Any unauthorized academic appropriation, plagiarism or act of claiming research priority will be held accountable in accordance with academic norms and relevant intellectual property laws and regulations.

## Disclaimer
This is an unpublished academic work. All contents in this repository are temporary review materials and do not represent the final published version. Contents may be adjusted, optimized and updated during the peer review process. The author shall not be liable for any unauthorized use or misinterpretation of the current resources by third parties.

2026.05.31

---

### Update: 2026.06.09
The module invocation method has been optimized. You can now import modules such as `physlainet` and execute them via `physlainet.run()`.
**Note**: All extracted files must be kept in the same directory as the original compressed package.
