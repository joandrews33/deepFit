import torch
from simulate.simulate_camera import get_camera_image
from simulate.simulate_camera import add_molecule


def simulate_test_data(n_pix=128, mols_per_frame=10, mean_photons=64, trim_px=21):

    X = get_camera_image(n_pix)
    X, molecule_list = add_molecule(X, mean_num_photons=mean_photons, trim_px=trim_px)
    for i in range(1, mols_per_frame):
        X, molecule_list = add_molecule(X, molecule_list, mean_num_photons=mean_photons, trim_px=trim_px)

    detection_map = torch.zeros(X.size())

    for position in molecule_list:
        indices = torch.cat((torch.zeros(2), torch.floor(position)), 0).long()
        detection_map[indices.split(1)] = 1

    #temp = torch.zeros(detection_map.size())
    #detection_map[:, :, trim_pix+1:-trim_pix-1, trim_pix+1:-trim_pix-1] = temp[:, :, trim_pix+1:-trim_pix-1, trim_pix+1:-trim_pix-1] #deleting detections from the boundary

    n_mols = torch.sum(detection_map).long() #molecule_list.size()[0]

    return X, detection_map, n_mols, molecule_list

