import torch


class DualSelfTeachingLoss:
    def __init__(self, beta=0.5, gamma=0.5, sigma=0.2, temperature=1.0, n_passages_per_query=8):
        """
        n_passages_per_query: number of passages per query (positive passage + hard negative passages), in our case is 8 since we have 7 hard negatives
        """
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.temperature = temperature
        self.n_passages_per_query = n_passages_per_query

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.KL = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, q_reps: torch.Tensor, p_reps: torch.Tensor, typo_q_reps: torch.Tensor, effective_bsz: int):
        """
        q_reps: original queries representations with shape (queries_num, embedding_dim)
        p_reps: passages representations with shape (passages_num, embedding_dim)
        typo_q_reps: misspelled queries representations with shape (queries_num x typos_num, embedding_dim)
        effective_bsz: batch size
        """
        ################################################## Similarity Score Calculation ##################################################
        # Passage retrieval (query-to-passages)
        # Original queries
        qp_scores = torch.matmul(q_reps, p_reps.transpose(0, 1))  # (queries_num, passages_num)
        qp_scores = qp_scores.view(effective_bsz, -1)  # (queries_num, passages_num)
        queries_num, passages_num = qp_scores.shape
        # Misspelled queries
        typo_qp_scores = torch.matmul(typo_q_reps, p_reps.transpose(0, 1))  # (queries_num x typos_num, passages_num)
        typo_qp_scores = typo_qp_scores.view(queries_num, -1, passages_num)  # (queries_num, typos_num, passages_num)
        typos_num = typo_qp_scores.shape[1]

        # Query retrieval (passage-to-queries)
        # Original queries
        pq_scores = torch.matmul(p_reps, q_reps.transpose(0, 1))  # (passages_num, queries_num)
        pq_scores = pq_scores.view(passages_num, queries_num)  # (passages_num, queries_num)
        # Remove hard negative passages
        pos_pq_scores = pq_scores[torch.arange(0, pq_scores.shape[0], self.n_passages_per_query),
                        :]  # (pos_passages_num, queries_num)
        # Misspelled queries
        typo_pq_scores = torch.matmul(p_reps, typo_q_reps.transpose(0, 1))  # (passages_num, queries_num x typos_num)
        typo_pq_scores = typo_pq_scores.view(passages_num, queries_num,
                                             typos_num)  # (passages_num, queries_num, typos_num)
        ######################################################## Loss Calculation ########################################################
        # Dual Cross-Entropy Loss
        # Passage Retrieval
        qp_target = torch.arange(
            qp_scores.size(0),
            device=qp_scores.device,
            dtype=torch.long
        )
        qp_target = qp_target * self.n_passages_per_query
        qp_ce_loss = self.cross_entropy(qp_scores, qp_target)
        # Query Retrieval
        pq_target = torch.arange(
            pos_pq_scores.size(0),
            device=pos_pq_scores.device,
            dtype=torch.long
        )
        pq_ce_loss = self.cross_entropy(pos_pq_scores, pq_target)
        ce_loss = (1 - self.gamma) * qp_ce_loss + self.gamma * pq_ce_loss

        # Dual KL-Divergence Loss
        # Passage Retrieval Consistency
        qp_kl_loss = 0.0
        for i in range(typos_num):
            qp_kl_loss += self.KL(
                self.log_softmax(typo_qp_scores[:, i, :]),
                self.log_softmax(qp_scores.detach() / self.temperature)
            ) / typos_num
        # Query Retrieval Consistency
        pq_kl_loss = 0.0
        for i in range(typos_num):
            pq_kl_loss += self.KL(
                self.log_softmax(typo_pq_scores[:, :, i]),
                self.log_softmax(pq_scores.detach() / self.temperature)
            ) / typos_num
        kl_loss = (1 - self.sigma) * qp_kl_loss + self.sigma * pq_kl_loss

        # Dual Self-Teaching Loss
        loss = (1 - self.beta) * ce_loss + self.beta * kl_loss
        return loss, qp_scores


class DualSelfTeachingMultiPositiveLoss:
    def __init__(self, beta=0.5, gamma=0.5, sigma=0.2, temperature=1.0, n_passages_per_query=8):
        """
        n_passages_per_query: number of passages per query (positive passage + hard negative passages), in our case is 8 since we have 7 hard negatives
        """
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.temperature = temperature
        self.n_passages_per_query = n_passages_per_query

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.KL = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def sum_log_nce_loss(self, logits, mask, n_pos, reduction='mean'):

        gold_scores = logits.masked_fill(~(mask.bool()), 0)
        gold_scores_sum = gold_scores.sum(-1)  

        neg_logits = logits.masked_fill(mask.bool(), float('-inf'))
        neg_log_sum_exp = torch.logsumexp(neg_logits, -1, keepdim=True)

        norm_term = torch.logaddexp(logits, neg_log_sum_exp).masked_fill(~(
            mask.bool()), 0).sum(-1)

        gold_log_probs = gold_scores_sum - norm_term
        gold_log_probs /= n_pos

        loss = -gold_log_probs.sum()

        if reduction == 'mean':
            loss /= logits.size(0)

        return loss

    def __call__(self, q_reps: torch.Tensor, p_reps: torch.Tensor, typo_q_reps: torch.Tensor, effective_bsz: int):
        """
        q_reps: original queries representations with shape (queries_num, embedding_dim)
        p_reps: passages representations with shape (passages_num, embedding_dim)
        typo_q_reps: misspelled queries representations with shape (queries_num x typos_num, embedding_dim)
        effective_bsz: batch size
        """
        ################################################## Similarity Score Calculation ##################################################
        # Passage retrieval (query-to-passages)
        # Original queries
        qp_scores = torch.matmul(q_reps, p_reps.transpose(0, 1))  # (queries_num, passages_num)
        qp_scores = qp_scores.view(effective_bsz, -1)  # (queries_num, passages_num)
        queries_num, passages_num = qp_scores.shape
        # Misspelled queries
        typo_qp_scores = torch.matmul(typo_q_reps, p_reps.transpose(0, 1))  # (queries_num x typos_num, passages_num)
        typo_qp_scores = typo_qp_scores.view(queries_num, -1, passages_num)  # (queries_num, typos_num, passages_num)
        typos_num = typo_qp_scores.shape[1]

        # Query retrieval (passage-to-queries)
        # Original queries
        pq_scores = torch.matmul(p_reps, q_reps.transpose(0, 1))  # (passages_num, queries_num)
        pq_scores = pq_scores.view(passages_num, queries_num)  # (passages_num, queries_num)
        # Remove hard negative passages
        pos_pq_scores = pq_scores[torch.arange(0, pq_scores.shape[0], self.n_passages_per_query),
                        :]  # (pos_passages_num, queries_num)
        # Misspelled queries
        typo_pq_scores = torch.matmul(p_reps, typo_q_reps.transpose(0, 1))  # (passages_num, queries_num x typos_num)
        typo_pq_scores = typo_pq_scores.view(passages_num, queries_num,
                                             typos_num)  # (passages_num, queries_num, typos_num)
        ######################################################## Loss Calculation ########################################################
        # Dual Cross-Entropy Loss
        # Passage Retrieval
        qp_target = torch.arange(
            qp_scores.size(0),
            device=qp_scores.device,
            dtype=torch.long
        )
        qp_target = qp_target * self.n_passages_per_query
        qp_ce_loss = self.cross_entropy(qp_scores, qp_target)

        ###################
        # Query Retrieval
        q_reps_all = torch.cat((q_reps, typo_q_reps), 0)

        # Original+Typo queries
        pq_scores_all = torch.matmul(p_reps,
                                     q_reps_all.transpose(0, 1))  # (passages_num, (queries_num*typos_num)+queries_num)
        pq_scores_all = pq_scores_all.view(passages_num, (
                queries_num * typos_num) + queries_num)  # (passages_num, (queries_num*typos_num)+queries_num)

        # Remove hard negative passages
        pos_pq_scores_all = pq_scores_all[torch.arange(0, pq_scores_all.shape[0], self.n_passages_per_query),
                            :]  # (pos_passages_num, (queries_num*typos_num)+queries_num)

        # original questions
        pq_target_1 = torch.arange(
            start=0,
            end=queries_num,
            device=pos_pq_scores_all.device,
            dtype=torch.long
        )
        # typoed questions
        pq_target_2 = torch.arange(
            start=queries_num,
            end=pos_pq_scores_all.size(1),
            step=typos_num,
            device=pos_pq_scores_all.device,
            dtype=torch.long
        )

        pq_target_1 = pq_target_1.cpu().detach().numpy()
        pq_target_2 = pq_target_2.cpu().detach().numpy()

        pq_masks = torch.zeros(pos_pq_scores_all.size(0), pos_pq_scores_all.size(1))

        for k, v in enumerate(pq_target_1):
            pq_masks[k, v] = 1

        for k, v in enumerate(pq_target_2):
            pq_masks[k, v:v + typos_num] = 1
        pq_masks = pq_masks.to(pos_pq_scores_all.device)

        pq_ce_loss = self.sum_log_nce_loss(pos_pq_scores_all, pq_masks, typos_num + 1, reduction='mean')
        ###################

        ce_loss = (1 - self.gamma) * qp_ce_loss + self.gamma * pq_ce_loss

        # Dual KL-Divergence Loss
        # Passage Retrieval Consistency
        qp_kl_loss = 0.0
        for i in range(typos_num):
            qp_kl_loss += self.KL(
                self.log_softmax(typo_qp_scores[:, i, :]),
                self.log_softmax(qp_scores.detach() / self.temperature)
            ) / typos_num
        # Query Retrieval Consistency
        pq_kl_loss = 0.0
        for i in range(typos_num):
            pq_kl_loss += self.KL(
                self.log_softmax(typo_pq_scores[:, :, i]),
                self.log_softmax(pq_scores.detach() / self.temperature)
            ) / typos_num
        kl_loss = (1 - self.sigma) * qp_kl_loss + self.sigma * pq_kl_loss

        # Dual Self-Teaching Loss
        loss = (1 - self.beta) * ce_loss + self.beta * kl_loss
        return loss, qp_scores

