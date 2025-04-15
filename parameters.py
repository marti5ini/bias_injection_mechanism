n_sample = 5000
parameter1 = 1
outcome = 'Y'
parameters2_list = [1, 1.2, 1.6, 2, 2.5, 3, 3.5]
cd_algo = 'pc'

A_Y = True
noise = 'exp'
y_binary = False
folder_name = 'journal'

mediators = ['mediator_' + str(i) for i in [0, 1, 2, 3]]
med_confounders = ['med_confounder_' + str(i) for i in [0, 1, 2, 3]]
confounders = ['confounder_' + str(i) for i in range(2)]
colliders = ['collider_' + str(i) for i in range(2)]
combined = {'mediators': mediators, 'mediators and confounders': med_confounders, 'confounders': confounders, 'colliders': colliders}
#combined = {'colliders': colliders}
naming_structures = dict()



