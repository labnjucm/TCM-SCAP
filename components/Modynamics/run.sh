(echo "6"; echo "5") | gmx pdb2gmx -f protein.pdb -o protein_processed.gro -p topol.top -ignh

gmx editconf -f protein_processed.gro -o protein_newbox.gro -c -d 1.0 -bt cubic
gmx solvate -cp protein_newbox.gro -cs spc216.gro -o protein_solv.gro -p topol.top
gmx grompp -f ions.mdp -c protein_solv.gro -p topol.top -o ions.tpr -maxwarn 6
echo "15" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral
gmx grompp -f em1.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 6
gmx mdrun -v -deffnm em
gmx grompp -f em2.mdp -c em.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em
echo "2" | gmx genrestr -f ligand.gro -o posre_ligand.itp -fc 1000 1000 1000
(echo "1 | 13"; echo "q") |gmx make_ndx -f em.gro -o index.ndx
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt.tpr
gmx mdrun -v -deffnm nvt -gpu_id 0
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx -o npt.tpr -maxwarn 6
gmx mdrun -v -deffnm npt -gpu_id 0
gmx grompp -f md.mdp -c npt.gro -r npt.gro -t npt.cpt -n index.ndx -o md_0_1.tpr -p topol.top -maxwarn 6
gmx mdrun -v -deffnm md_0_1 -gpu_id 0
