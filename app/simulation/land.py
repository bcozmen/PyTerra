import numpy as np

from scipy.ndimage import binary_dilation

class Market:
    def __init__(self, name):
        self.name = name
        self.prices = {} # dict of good_name: price

class City: 
    def __init__(self, id = 1, maps = None, pos = (0,0)):
        self.id = id
        self.name = f"City {id}"
        self.pos = pos
        self.cities = []
        self.agents = []

        self.maps = maps #3d int array (x,y, [city_id, type])
        self.city_register = {}
        self.register_pos(pos, 1) #register initial position with land type 1 (urban)
        self.calculate_growth_trajectory()

    def register_pos(self, pos, land_type = 0):
        if type(pos) == tuple:
            pos = [pos]

        for p in pos:
            self.maps["city"][p[0], p[1], 0] = self.id
            self.maps["city"][p[0], p[1], 1] = land_type #owner id, for now just 0
            self.city_register[p] = self.id

    def calculate_growth_trajectory(self):
        fertility = self.maps["fertility"] 
        distance_map = np.sqrt((np.arange(fertility.shape[0])[:, None] - self.pos[0])**2 + (np.arange(fertility.shape[1])[None, :] - self.pos[1])**2)
        distance_map = 1 - (distance_map / np.max(distance_map))
        distance_map = distance_map ** 2  # give more weight to closer cells

        growth_score = fertility * distance_map

        #not in the sea or river
        growth_score[ self.maps["sea"] | self.maps["river"] ] = -np.inf
        self.growth_score = growth_score

    def grow_urban(self, amount = 1):
        #get urban land of this city
        urban_mask = self.maps["city"][:,:,1] == 1
        city_mask = self.maps["city"][:,:,0] == self.id
        urban_mask = urban_mask & ~city_mask
        

    def grow(self, amount = 1):
        #delete city from growth scores
        self.growth_score[self.maps["city"][:,:,0] > 0] = -np.inf

        #find the N amount of cells with the highest growth score efficiently    
        flat_indices = np.argpartition(self.growth_score.flatten(), -amount)[-amount:]
        max_cell_indices = np.unravel_index(flat_indices, self.growth_score.shape)

        #make a list of the new positions to grow into and register them
        new_positions = list(zip(max_cell_indices[0], max_cell_indices[1]))
        self.register_pos(new_positions)    
        self._update_position()
        self.calculate_growth_trajectory() #recalculate growth trajectory after growing

    def _calculate_growth_mask(self):
        city_mask = self.maps["city"][:,:,0] == self.id
        growth_mask = binary_dilation(city_mask) & ~city_mask

        #cant grow into sea or river or other cities
        growth_mask &= ~(self.maps["sea"] | self.maps["river"] | (self.maps["city"][:,:,0] > 0))

        return growth_mask

    def _update_position(self):
        #update position to the center of mass of the city
        city_mask = self.maps["city"][:,:,0] == self.id
        if np.sum(city_mask) > 0:
            self.pos = np.mean(np.argwhere(city_mask), axis=0).astype(int)
    

        


class Land:
    def __init__(self, name):
        pass
class RuralLand(Land):
    def __init__(self, name):
        super().__init__(name)

class SuburbanLand(Land):
    def __init__(self, name):
        super().__init__(name)

class UrbanLand(Land):
    def __init__(self, name):
        super().__init__(name)
