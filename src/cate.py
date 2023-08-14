from .graph_model import GraphModel
import torch.optim as optim
import torch as th
import torch.nn as nn
from tqdm import tqdm, trange
import logging

class CatE(GraphModel):
    def __init__(self,
                 use_case,
                 *args,
                 **kwargs):
        super().__init__(use_case, "cat", *args, **kwargs)

        
                                   
    def train(self):
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
                                                                                    
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        min_lr = self.lr/10
        max_lr = self.lr
        print("Min lr: {}, Max lr: {}".format(min_lr, max_lr))
        
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr,
                                                   max_lr=max_lr, step_size_up = 20,
                                                   cycle_momentum = False)

        criterion_bpr = nn.LogSigmoid()
        
        self.model = self.model.to(self.device)

        train_graph_dataloader = self.create_graph_dataloader(mode="train", batch_size=self.batch_size)

        tolerance = 0
        best_loss = float("inf")
        best_mr = float("inf")
        classes_ids = th.tensor(list(self.classes_ids), dtype=th.long,
                                     device=self.device)

        for epoch in trange(self.epochs, desc=f"Training..."):
            logging.info(f"Epoch: {epoch+1}")
            self.model.train()

            graph_loss = 0
            for head, rel, tail in tqdm(train_graph_dataloader, desc="Processing batches"):
                head = head.to(self.device)
                rel = rel.to(self.device)
                tail = tail.to(self.device)
                
                data = (head, rel, tail)
                pos_logits = self.model.forward(data)

                neg_logits = 0
                for i in range(self.num_negs):
                    neg_tail = th.randint(0, len(self.node_to_id), (len(head),), device=self.device)
                    data = (head, rel, neg_tail)
                    neg_logits += self.model.forward(data)
                                                            
                neg_logits /= (self.num_negs)

                batch_loss = -criterion_bpr(pos_logits - neg_logits - self.margin).mean()
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()

            graph_loss /= len(train_graph_dataloader)

            
            valid_subsumption_mr = self.compute_ranking_metrics(mode="valid_subsumption")
            valid_membership_mr = self.compute_ranking_metrics(mode="valid_membership")
            total_mr = (valid_subsumption_mr + valid_membership_mr)
            sub_weight = 0.5 #valid_subsumption_mr/total_mr
            mem_weight = 0.5 #valid_membership_mr/total_mr
            valid_mr = sub_weight*valid_subsumption_mr + mem_weight*valid_membership_mr
            
            if valid_mr < best_mr:
                best_mr = valid_mr
                th.save(self.model.state_dict(), self.model_path)
                tolerance = self.initial_tolerance+1
                print("Model saved")
            print(f"Training loss: {graph_loss:.6f}\tValidation mean_ranks: sub-{valid_subsumption_mr:.6f}, mem-{valid_membership_mr:.6f}, avg-{valid_mr:.6f}")

            tolerance -= 1
            if tolerance == 0:
                print("Early stopping")
                break

        self.save_embeddings_data()        

                                             
