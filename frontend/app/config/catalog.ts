import type { Catalog } from "../lib/types";

// 直接访问 Gradio 应用（不使用 Nginx 反向代理）
const D = process.env.NEXT_PUBLIC_DOCKING_PATH || "http://0.0.0.0:7861";
const M = process.env.NEXT_PUBLIC_MD_PATH || "http://0.0.0.0:7862";
const O = process.env.NEXT_PUBLIC_ORCA_PATH || "http://0.0.0.0:7863";

const md = (s: string) => s.trim();

export const catalog: Catalog = {
  sections: [
    {
      title: "获取数据",
      items: [
        {
          key: "rcsb",
          title: "RCSB PDB",
          intro: md(`**RCSB PDB**：全球最权威的蛋白质三维结构数据库，包含超过 20 万个实验解析的生物大分子结构。支持 PDB/mmCIF 格式下载，提供 REST/GraphQL API。`),
          link: "https://www.rcsb.org/",
          detailsSlug: "rcsb-pdb"
        },
        {
          key: "alphafold",
          title: "AlphaFold DB",
          intro: md(`**AlphaFold DB**：由 DeepMind/EMBL-EBI 提供的 AI 预测蛋白质结构数据库，覆盖超过 2 亿个蛋白质。当 PDB 中无实验结构时的首选。`),
          link: "https://alphafold.ebi.ac.uk/",
          detailsSlug: "alphafold-db"
        },
        {
          key: "chembl",
          title: "ChEMBL",
          intro: md(`**ChEMBL**：开放化合物生物活性数据库，包含 220 万+化合物和 1900 万+活性数据点。提供 Python API 和 REST 接口。`),
          link: "https://www.ebi.ac.uk/chembl/",
          detailsSlug: "chembl"
        },
        {
          key: "zinc",
          title: "ZINC",
          intro: md(`**ZINC**：用于虚拟筛选的大型购买化合物库，包含数十亿个可购买化合物。支持子结构/相似性搜索和批量下载。`),
          link: "https://zinc.docking.org/",
          detailsSlug: "zinc"
        },
        {
          key: "pubchem",
          title: "PubChem",
          intro: md(`**PubChem**：NCBI 提供的公共化学数据库，含 1.1 亿+化合物。提供 PUG-REST API，支持 SMILES/CID/性质查询。`),
          link: "https://pubchem.ncbi.nlm.nih.gov/",
          detailsSlug: "pubchem"
        }
      ]
    },
    {
      title: "分子对接",
      items: [
        {
          key: "vina",
          title: "AutoDock Vina",
          intro: md(`**Vina**：最流行的开源分子对接引擎，简单高效，广泛应用于虚拟筛选和药物设计。`),
          link: "http://vina.scripps.edu/",
          detailsSlug: "autodock-vina"
        },
        {
          key: "smina",
          title: "Smina",
          intro: md(`**Smina**：Vina 的改进版本，提供更多评分函数选项和灵活的搜索策略。`),
          link: "https://github.com/mwojcikowski/smina",
          detailsSlug: "smina"
        },
        {
          key: "gnina",
          title: "GNINA",
          intro: md(`**GNINA**：整合深度学习评分函数的对接工具，使用 CNN 评估蛋白-配体相互作用。`),
          link: "https://github.com/gnina/gnina",
          detailsSlug: "gnina"
        },
        {
          key: "open-gradio-docking",
          title: "打开本地分子对接界面",
          intro: md(`直接访问本地对接界面（\`${D}\`）。`),
          iframeSrc: D,
          requires: "docking"
        }
      ]
    },
    {
      title: "分子动力学模拟",
      items: [
        {
          key: "gromacs",
          title: "GROMACS",
          intro: md(`**GROMACS**：高性能分子动力学软件包，支持多种硬件与力场，广泛用于蛋白质、膜系统模拟。`),
          link: "https://www.gromacs.org/",
          detailsSlug: "gromacs"
        },
        {
          key: "openmm",
          title: "OpenMM",
          intro: md(`**OpenMM**：基于 GPU 加速的 MD 库，Python 友好，支持自定义力场和高度可扩展。`),
          link: "http://openmm.org/",
          detailsSlug: "openmm"
        },
        {
          key: "namd",
          title: "NAMD",
          intro: md(`**NAMD**：并行分子动力学引擎，适合大规模生物体系模拟，CHARMM 力场兼容性好。`),
          link: "https://www.ks.uiuc.edu/Research/namd/",
          detailsSlug: "namd"
        },
        {
          key: "open-gradio-md",
          title: "打开分子动力学界面",
          intro: md(`直接访问本地分子动力学界面（\`${M}\`）`),
          iframeSrc: M,
          requires: "md"
        }        
      ]
    },
    {
      title: "ADMET 分析",
      items: [
        {
          key: "swissadme",
          title: "SwissADME",
          intro: md(`
**SwissADME**：免费的 ADME 预测工具，评估药物相似性、生物利用度、药代动力学性质等。

提供多种预测功能：
- Lipinski 五规则
- 血脑屏障通透性
- P糖蛋白底物
- CYP450 酶抑制
- 药物相似性雷达图
          `),
          link: "https://www.swissadme.ch/",
          detailsSlug: "swissadme"
        },
        {
          key: "preadmet",
          title: "PreADMET",
          intro: md(`
**PreADMET**：ADMET（吸收、分布、代谢、排泄、毒性）预测平台。

提供全面的药物性质预测：
- 吸收性质（HIA, Caco-2）
- 分布性质（BBB, PPB）
- 代谢预测（CYP 底物/抑制）
- 毒性预测（hERG, Ames）
          `),
          link: "https://preadmet.webservice.bmdrc.org/",
          detailsSlug: "preadmet"
        }
      ]
    },
    {
      title: "计算化学分析",
      items: [
        {
          key: "orca",
          title: "ORCA",
          intro: md(`**ORCA**：现代量子化学程序，支持 DFT/MP2/CC 等方法，适合中大型分子计算。`),
          link: "https://orcaforum.kofo.mpg.de/",
          detailsSlug: "orca"
        },
        {
          key: "gaussian",
          title: "Gaussian",
          intro: md(`**Gaussian**：经典量化化学软件，覆盖广泛计算方法，拥有庞大用户社区。`),
          link: "https://gaussian.com/",
          detailsSlug: "gaussian"
        },
        {
          key: "qchem",
          title: "Q-Chem",
          intro: md(`**Q-Chem**：现代量化化学程序，提供众多前沿功能和高效并行计算。`),
          link: "https://www.q-chem.com/",
          detailsSlug: "q-chem"
        },
        {
          key: "open-gradio-orca",
          title: "打开ORCA 界面",
          intro: md(`直接访问本地 ORCA 量化计算界面（\`${O}\`）`),
          iframeSrc: O,
          requires: "orca"
        }
        
      ]
    }
  ]
};
