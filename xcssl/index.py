import numpy as np


class PhenotypeIndex:
    def __init__(self, encoding, rasterizer, phenotypes):
        self._encoding = encoding
        self._rasterizer = rasterizer

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        (self._phenotype_grid_cells_map,
         self._grid_cell_phenotypes_map) = self._gen_maps(
             self._encoding,
             self._rasterizer,
             phenotype_set=self._phenotype_count_map.keys())

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _gen_maps(self, encoding, rasterizer, phenotype_set):

        phenotype_grid_cells_map = {}
        grid_cell_phenotypes_map = np.empty(shape=rasterizer.num_grid_cells,
                                            dtype="object")
        for idx in range(0, rasterizer.num_grid_cells):
            grid_cell_phenotypes_map[idx] = set()

        for phenotype in phenotype_set:

            grid_cells_covered = rasterizer.rasterize_phenotype(
                encoding, phenotype)
            phenotype_grid_cells_map[phenotype] = grid_cells_covered

            for grid_cell in grid_cells_covered:
                (grid_cell_phenotypes_map[grid_cell]).add(phenotype)

        return (phenotype_grid_cells_map, grid_cell_phenotypes_map)

    def gen_sparse_phenotype_matching_map(self, obs):
        sparse_phenotype_matching_map = {}

        grid_cell = self._rasterizer.rasterize_obs(obs)

        for phenotype in self._grid_cell_phenotypes_map[grid_cell]:

            sparse_phenotype_matching_map[
                phenotype] = self._encoding.does_phenotype_match(
                    phenotype, obs)

        return sparse_phenotype_matching_map

    def gen_matching_trace(self, obs):

        sparse_phenotype_matching_map = {}

        grid_cell = self._rasterizer.rasterize_obs(obs)

        phenotype_set = self._grid_cell_phenotypes_map[grid_cell]

        for phenotype in phenotype_set:

            sparse_phenotype_matching_map[
                phenotype] = self._encoding.does_phenotype_match(
                    phenotype, obs)

        num_matching_ops_done = len(phenotype_set)

        return (sparse_phenotype_matching_map, num_matching_ops_done)

    def gen_partial_matching_trace(self, obs):
        grid_cell = self._rasterizer.rasterize_obs(obs)
        return len(self._grid_cell_phenotypes_map[grid_cell])

    def try_add_phenotype(self, phenotype):
        try:
            self._phenotype_count_map[phenotype] += 1
        except KeyError:
            self._phenotype_count_map[phenotype] = 1
            do_add = True
        else:
            do_add = False

        if do_add:
            self._add_phenotype(phenotype)

    def _add_phenotype(self, addee):
        grid_cells_covered = self._rasterizer.rasterize_phenotype(
            self._encoding, addee)
        self._phenotype_grid_cells_map[addee] = grid_cells_covered

        for grid_cell in grid_cells_covered:
            (self._grid_cell_phenotypes_map[grid_cell]).add(addee)

    def try_remove_phenotype(self, phenotype):
        count = self._phenotype_count_map[phenotype]
        count -= 1

        do_remove = (count == 0)

        if do_remove:
            del self._phenotype_count_map[phenotype]
            self._remove_phenotype(phenotype)

        else:
            self._phenotype_count_map[phenotype] = count

    def _remove_phenotype(self, removee):
        grid_cells_covered = self._phenotype_grid_cells_map[removee]

        for grid_cell in grid_cells_covered:
            (self._grid_cell_phenotypes_map[grid_cell]).remove(removee)

        del self._phenotype_grid_cells_map[removee]
