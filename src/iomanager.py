from pathlib import Path
from typing import Dict
import numpy as np
import h5py
import src.params as params 
from src.varindexes import IR, IU, IV, IW, IP, IBX, IBY, IBZ, IPSI


class IOManager:
    def __init__(self, outname: str = "run", dirname: str = "data"):
        self.outname = outname
        self.dirname = Path(dirname)
        self.ite_nzeros = 4  # Format des itérations : ite_0000
        self.setup_dirdata()

        # Récupération des paramètres depuis src.params
        self.Nx = params.Nx
        self.Ny = params.Ny
        self.Ntx = params.Ntx
        self.Nty = params.Nty
        self.ibeg = params.ibeg
        self.iend = params.iend
        self.jbeg = params.jbeg
        self.jend = params.jend
        self.dx = params.dx
        self.dy = params.dy
        self.xmin = params.xmin
        self.ymin = params.ymin
        self.MHD = params.MHD
        self.write_ghost_cells = getattr(params, 'write_ghost_cells', False)

    def setup_dirdata(self) -> None:
        """Crée le dossier de sortie s'il n'existe pas."""
        self.dirname.mkdir(parents=True, exist_ok=True)

    def save_solution(self, Q: np.ndarray, iteration: int, t: float, unique_output: bool = False) -> None:
        """
        Sauvegarde la solution dans un fichier HDF5 et génère un XMF.
        Args:
            Q: Tableau 3D (Ny, Nx, Nfields) contenant les variables.
            iteration: Numéro de l'itération.
            t: Temps de simulation.
            unique_output: Si True, utilise un seul fichier HDF5 pour toutes les itérations.
        """
        if unique_output:
            self._save_solution_unique(Q, iteration, t)
        else:
            self._save_solution_multiple(Q, iteration, t)

    def _save_solution_multiple(self, Q: np.ndarray, iteration: int, t: float) -> None:
        """Sauvegarde une solution par fichier (comme dans ton code C++)."""
        iteration_str = f"ite_{iteration:0{self.ite_nzeros}d}"
        h5_filename = self.dirname / f"{iteration_str}.h5"
        xmf_filename = self.dirname / f"{iteration_str}.xmf"

        # Détermine les bornes (avec ou sans ghost cells)
        j0, jN = self.jbeg, self.jend
        i0, iN = self.ibeg, self.iend
        if self.write_ghost_cells:
            j0, jN = 0, self.Nty
            i0, iN = 0, self.Ntx

        # Dimensions pour XDMF (fdim = cellules, gdim = sommets)
        fNy, fNx = jN - j0, iN - i0  # Nombre de cellules
        gNy, gNx = fNy + 1, fNx + 1  # Nombre de sommets

        # Coordonnées (sommets)
        x_coords = np.zeros(gNx * gNy)
        y_coords = np.zeros(gNx * gNy)
        idx = 0
        for j in range(j0, jN + 1):
            for i in range(i0, iN + 1):
                x_coords[idx] = (i - self.ibeg) * self.dx + self.xmin
                y_coords[idx] = (j - self.jbeg) * self.dy + self.ymin
                idx += 1

        # Sauvegarde HDF5
        with h5py.File(h5_filename, "w") as f:
            # Attributs globaux
            f.attrs["time"] = t
            f.attrs["iteration"] = iteration
            f.attrs["Nx"] = self.Nx
            f.attrs["Ny"] = self.Ny
            f.attrs["Ntx"] = self.Ntx
            f.attrs["Nty"] = self.Nty
            f.attrs["ibeg"] = self.ibeg
            f.attrs["iend"] = self.iend
            f.attrs["jbeg"] = self.jbeg
            f.attrs["jend"] = self.jend
            f.attrs["problem"] = params.problem_name
            f.attrs["dx"] = self.dx
            f.attrs["dy"] = self.dy
            f.attrs["xmin"] = self.xmin
            f.attrs["ymin"] = self.ymin

            # Coordonnées
            f.create_dataset("x", data=x_coords)
            f.create_dataset("y", data=y_coords)

            # Extraction des champs (cellules)
            trho = np.zeros(fNy * fNx)
            tu = np.zeros(fNy * fNx)
            tv = np.zeros(fNy * fNx)
            tprs = np.zeros(fNy * fNx)
            idx = 0
            for j in range(j0, jN):
                for i in range(i0, iN):
                    trho[idx] = Q[i, j, IR]
                    tu[idx] = Q[i, j, IU]
                    tv[idx] = Q[i, j, IV]
                    tprs[idx] = Q[i, j, IP]
                    idx += 1

            f.create_dataset("rho", data=trho)
            f.create_dataset("u", data=tu)
            f.create_dataset("v", data=tv)
            f.create_dataset("prs", data=tprs)

            # Champs MHD (si activé)
            # if self.MHD:
            tw = np.zeros(fNy * fNx)
            tbx = np.zeros(fNy * fNx)
            tby = np.zeros(fNy * fNx)
            tbz = np.zeros(fNy * fNx)
            tpsi = np.zeros(fNy * fNx)
            idx = 0
            for j in range(j0, jN):
                for i in range(i0, iN):
                    tw[idx] = Q[i, j, IW]
                    tbx[idx] = Q[i, j, IBX]
                    tby[idx] = Q[i, j, IBY]
                    tbz[idx] = Q[i, j, IBZ]
                    tpsi[idx] = Q[i, j, IPSI]
                    idx += 1
            f.create_dataset("w", data=tw)
            f.create_dataset("bx", data=tbx)
            f.create_dataset("by", data=tby)
            f.create_dataset("bz", data=tbz)
            f.create_dataset("psi", data=tpsi)

        # Génération du fichier XMF avec le bon nom de fichier HDF5
        self._generate_xmf(h5_filename, xmf_filename, t, fNy, fNx, gNy, gNx, iteration_str)

    def _generate_xmf(self, h5_filename: Path, xmf_filename: Path, t: float,
                      fNy: int, fNx: int, gNy: int, gNx: int, iteration_str: str) -> None:
        """Génère un fichier XMF compatible avec ParaView, avec le bon nom de fichier HDF5."""
        # Le nom du fichier HDF5 dans le XMF est juste le nom du fichier, sans chemin
        h5_basename = h5_filename.name

        with open(xmf_filename, "w") as xdmf_fd:
            xdmf_fd.write(f'''<?xml version="1.0" ?>
                <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [
                <!ENTITY file "{h5_basename}:">
                <!ENTITY fdim "{fNy} {fNx}">
                <!ENTITY gdim "{gNy} {gNx}">
                <!ENTITY GridEntity '
                <Topology TopologyType="2DSMesh" Dimensions="&gdim;"/>
                <Geometry GeometryType="X_Y">
                <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/x</DataItem>
                <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/y</DataItem>
                </Geometry>'>
                ]>
                <Xdmf Version="3.0">
                <Domain>
                <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
                    <Grid Name="{iteration_str}" GridType="Uniform">
                    <Time Value="{t}" />
                    &GridEntity;
                ''')

            # Champs scalaires
            for field in ["rho", "prs"]:
                xdmf_fd.write(f'''
                    <Attribute Name="{field}" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{field}</DataItem>
                    </Attribute>
                ''')

            # Champs vectoriels
            if not self.MHD:
                xdmf_fd.write('''
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem Dimensions="&fdim; 2" ItemType="Function" Function="JOIN($0, $1)">
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/u</DataItem>
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/v</DataItem>
                        </DataItem>
                    </Attribute>
                ''')
            else:
                xdmf_fd.write('''
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem Dimensions="&fdim; 3" ItemType="Function" Function="JOIN($0, $1, $2)">
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/u</DataItem>
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/v</DataItem>
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/w</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="magnetic" AttributeType="Vector" Center="Cell">
                        <DataItem Dimensions="&fdim; 3" ItemType="Function" Function="JOIN($0, $1, $2)">
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/bx</DataItem>
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/by</DataItem>
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/bz</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="psi" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/psi</DataItem>
                    </Attribute>
                ''')

            xdmf_fd.write('''
                </Grid>
                </Grid>
            </Domain>
            </Xdmf>
            ''')

    def _save_solution_unique(self, Q: np.ndarray, iteration: int, t: float) -> None:
        """Sauvegarde dans un seul fichier HDF5 (groupes par itération)."""
        h5_filename = self.dirname / f"{self.outname}.h5"
        xmf_filename = self.dirname / f"{self.outname}.xmf"
        iteration_str = f"ite_{iteration:0{self.ite_nzeros}d}"

        # Détermine les bornes (avec ou sans ghost cells)
        j0, jN = self.jbeg, self.jend
        i0, iN = self.ibeg, self.iend
        if self.write_ghost_cells:
            j0, jN = 0, self.Nty
            i0, iN = 0, self.Ntx

        # Dimensions pour XDMF
        fNy, fNx = jN - j0, iN - i0
        gNy, gNx = fNy + 1, fNx + 1

        # Coordonnées (sommets)
        x_coords = np.zeros(gNx * gNy)
        y_coords = np.zeros(gNx * gNy)
        idx = 0
        for j in range(j0, jN + 1):
            for i in range(i0, iN + 1):
                x_coords[idx] = (i - self.ibeg) * self.dx + self.xmin
                y_coords[idx] = (j - self.jbeg) * self.dy + self.ymin
                idx += 1

        # Ouverture du fichier HDF5
        mode = "w" if iteration == 0 else "r+"
        with h5py.File(h5_filename, mode) as f:
            if iteration == 0:
                # Attributs globaux (première itération)
                f.attrs["Nx"] = self.Nx
                f.attrs["Ny"] = self.Ny
                f.attrs["Ntx"] = self.Ntx
                f.attrs["Nty"] = self.Nty
                f.attrs["problem"] = getattr(params, 'problem_name', 'unknown')
                f.create_dataset("x", data=x_coords)
                f.create_dataset("y", data=y_coords)

                # Header XMF
                with open(xmf_filename, "w") as xdmf_fd:
                    xdmf_fd.write(f'''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [
<!ENTITY file "{h5_filename.name}:">
<!ENTITY fdim "{fNy} {fNx}">
<!ENTITY gdim "{gNy} {gNx}">
<!ENTITY GridEntity '
<Topology TopologyType="2DSMesh" Dimensions="&gdim;"/>
<Geometry GeometryType="X_Y">
  <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/x</DataItem>
  <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/y</DataItem>
</Geometry>'>
]>
<Xdmf Version="3.0">
<Domain>
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
''')

            # Groupe pour cette itération
            ite_group = f.create_group(iteration_str)
            ite_group.attrs["time"] = t
            ite_group.attrs["iteration"] = iteration

            # Extraction des champs
            trho = np.zeros(fNy * fNx)
            tu = np.zeros(fNy * fNx)
            tv = np.zeros(fNy * fNx)
            tprs = np.zeros(fNy * fNx)
            idx = 0
            for j in range(j0, jN):
                for i in range(i0, iN):
                    trho[idx] = Q[i, j, IR]
                    tu[idx] = Q[i, j, IU]
                    tv[idx] = Q[i, j, IV]
                    tprs[idx] = Q[i, j, IP]
                    idx += 1

            ite_group.create_dataset("rho", data=trho)
            ite_group.create_dataset("u", data=tu)
            ite_group.create_dataset("v", data=tv)
            ite_group.create_dataset("prs", data=tprs)

            # Champs MHD
            # if self.MHD:
            tw = np.zeros(fNy * fNx)
            tbx = np.zeros(fNy * fNx)
            tby = np.zeros(fNy * fNx)
            tbz = np.zeros(fNy * fNx)
            tpsi = np.zeros(fNy * fNx)
            idx = 0
            for j in range(j0, jN):
                for i in range(i0, iN):
                    tw[idx] = Q[i, j, IW]
                    tbx[idx] = Q[i, j, IBX]
                    tby[idx] = Q[i, j, IBY]
                    tbz[idx] = Q[i, j, IBZ]
                    tpsi[idx] = Q[i, j, IPSI]
                    idx += 1
            ite_group.create_dataset("w", data=tw)
            ite_group.create_dataset("bx", data=tbx)
            ite_group.create_dataset("by", data=tby)
            ite_group.create_dataset("bz", data=tbz)
            ite_group.create_dataset("psi", data=tpsi)

            # Mise à jour du XMF
            with open(xmf_filename, "a") as xdmf_fd:
                xdmf_fd.write(f'''
    <Grid Name="{iteration_str}" GridType="Uniform">
      <Time Value="{t}" />
      &GridEntity;
''')

                # Champs scalaires
                for field in ["rho", "prs"]:
                    xdmf_fd.write(f'''
      <Attribute Name="{field}" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/{field}</DataItem>
      </Attribute>
''')

                # Champs vectoriels
                if not self.MHD:
                    xdmf_fd.write(f'''
      <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="&fdim; 2" ItemType="Function" Function="JOIN($0, $1)">
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/u</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/v</DataItem>
        </DataItem>
      </Attribute>
''')
                else:
                    xdmf_fd.write(f'''
      <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="&fdim; 3" ItemType="Function" Function="JOIN($0, $1, $2)">
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/u</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/v</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/w</DataItem>
        </DataItem>
      </Attribute>
      <Attribute Name="magnetic" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="&fdim; 3" ItemType="Function" Function="JOIN($0, $1, $2)">
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/bx</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/by</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/bz</DataItem>
        </DataItem>
      </Attribute>
      <Attribute Name="psi" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/{iteration_str}/psi</DataItem>
      </Attribute>
''')

                xdmf_fd.write('''
      </Grid>
''')

            # Footer XMF (à ajouter à la fin de la simulation)
            if iteration == 0:
                xdmf_fd.write('''
  </Grid>
</Domain>
</Xdmf>
''')

    def load_solution(self, iteration: int) -> Dict[str, np.ndarray]:
        """Charge une solution sauvegardée."""
        if getattr(params, 'multiple_outputs', False):
            h5_filename = self.dirname / f"ite_{iteration:0{self.ite_nzeros}d}.h5"
        else:
            h5_filename = self.dirname / f"{self.outname}.h5"

        if not h5_filename.exists():
            raise FileNotFoundError(f"Fichier {h5_filename} introuvable.")

        data = {}
        with h5py.File(h5_filename, "r") as f:
            if getattr(params, 'multiple_outputs', False):
                group = "/"
            else:
                group = f"ite_{iteration:0{self.ite_nzeros}d}/"

            # Chargement des champs
            for field in ["rho", "u", "v", "prs"]:
                data[field] = f[f"{group}{field}"][:]

            if self.MHD:
                for field in ["w", "bx", "by", "bz", "psi"]:
                    data[field] = f[f"{group}{field}"][:]

            # Chargement des attributs
            if getattr(params, 'multiple_outputs', False):
                data["time"] = f.attrs["time"]
            else:
                data["time"] = f[f"{group}"].attrs["time"]

        return data
