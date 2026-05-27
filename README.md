# MSc Dissertation — Decision Transformer for Off-Road Autonomous Navigation

**Student:** Rutuja Patil  
**Supervisor:** Tasmina Islam  
**Module:** 7CCSMPRJ — MSc Individual Project  
**Institution:** King's College London  
**Year:** 2025–2026  

## Project Overview

This project investigates the use of Decision Transformers (DT) for trajectory 
prediction and path planning in unstructured off-road environments. The DT model 
is benchmarked against LSTM and imitation learning baselines using data collected 
from the CARLA simulator (Town07 off-road map).

## Repository Structure

| Folder | Contents |
|--------|----------|
| `data/` | CARLA collected episodes — (state, action, reward, timestep) tuples |
| `models/` | Decision Transformer implementation and saved checkpoints |
| `baselines/` | LSTM and behavioural cloning baseline implementations |
| `eval/` | Evaluation scripts, metrics, result plots |
| `notebooks/` | Colab notebooks for data collection, training, evaluation |
| `dissertation/` | Dissertation LaTeX or Word source files |

## Dissertation Sections

- [ ] Introduction
- [ ] Literature Review
- [ ] Specification & Design
- [ ] Implementation
- [ ] Evaluation
- [ ] Ethics & Professional Issues
- [ ] Conclusion

## Key Technologies

- CARLA Simulator 0.9.16
- PyTorch
- Decision Transformer (Chen et al., 2021)
- Python 3.10

## Progress Log

| Phase | Activity | Status |
|-------|----------|--------|
| Foundation | CARLA 0.9.16 downloaded, extracted, Python client installed | ✅ Complete |
| Foundation | Literature review - all areas covered | ✅ Complete |
| Foundation | Preliminary project report submitted (30 Apr) | ✅ Complete |
| Foundation | Core paper study - all 3 papers | ✅ Complete |
| Foundation | HPC access requested | ✅ Complete |
| Implementation | CARLA data collection pipeline | ⏳ Pending HPC |
| Implementation | Decision Transformer + baselines | ⏳ Upcoming |
| Evaluation | Full evaluation suite | ⏳ Upcoming |
| Writing | Dissertation + video submission (6 Aug) | ⏳ Upcoming |
