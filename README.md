# Synthetic Travelers: Using LLMs to Generate Realistic Mobility Behaviours

This repository contains the code developed for this thesis, which investigates how well a pre-trained LLM can simulate realistic travel diaries from socio-demographic and spatial inputs, and evaluates the transferability of the proposed framework. The full document is available here: [thesis (PDF)](Syntrav_thesis/master_thesis_anacaleiro_20240696.pdf)

For any questions about this work, feel free to contact me at: 20240696@novaims.unl.pt

## File Structure

```
Thesis_SynTrav/
├── Notebooks/                                # Main pipeline, run in order
│   ├── 1_SynTravel_EDA_personas.ipynb        # Exploratory data analysis & persona construction
│   ├── 2_SynTravel_patterns.ipynb            # Activity pattern extraction (chain-of-thought)
│   ├── 3_SynTravel_generation.ipynb          # LLM travel diary generation
│   ├── 4_SynTravel_evaluation.ipynb          # Evaluation against ground truth
│   └── 5_SynTravel_Transferability.ipynb     # Transferability to Portugal (rural/urban)
│
├── Helpers/                    
│   ├── personas_functions.py       
│   ├── persona_scoring_validation.py
│   ├── cot_patterns.py             
│   ├── trajectory_generation*.py   
│   ├── poi_allocator.py            
│   ├── evaluation.py / variance_evaluation.py
│   ├── atypical_travelers.py
│   └── visualizations/            
│
├── prompt_template/       
│   ├── cot_prompt.py
│   ├── generation_prompt.py            
│   ├── generation_prompt_distance_ablation.py   
│   ├── geration_prompt_mode_ablation.py              
│   ├── geration_prompt_mode_distance_ablation.py  
│               
├── llm_config/
│   ├── llm_config.py                   
│
├── Json_files/         
│   ├── pt_ablations
│   ├── variance             
│  
├── human_evaluation_survey/     
│
├── smp/                         # Baseline necessary py files and analysis
│  
├── figures/     
│
├── Syntrav_thesis/              
```
