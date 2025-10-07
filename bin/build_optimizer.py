import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup

from optim.adam_sf_ref import AdamWScheduleFreeReference
from optim.adina import Adina
from optim.adina_sf_ref import AdinaSchedulefreeReference
from optim.adina_sf_approx import AdinaSchedulefreeApprox
from optim.adina_sf_lrc import AdinaSchedulefreeLRC

class BuildOptimizer():

    def __init__(
            self,
            model,
            lr,
            total_epochs,
            steps_per_epoch
            ):
        self.model = model
        self.lr = lr
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch


    # =================
    # ===== Adam ======
    # =================

    # 1. Adam (No Scheduler)
    def make_adam(self):
        opt = optim.Adam(
            self.model.parameters(), 
            lr=self.lr
            )
        return opt, None, "None"

    # 2. Adam + StepLR
    def make_step(self):
        opt = optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
            )
        sched = lr_scheduler.StepLR(
            opt, 
            step_size=5, 
            gamma=0.1
            )
        return opt, sched, "Step"

    # 3. Adam + MultiStepLR
    def make_multistep(self):
        opt = optim.Adam(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = lr_scheduler.MultiStepLR(
            opt, milestones=[5, 10, 15], 
            gamma=0.1
            )
        return opt, sched, "MultiStep"

    # 4. Adam + ExponentialLR
    def make_exponential(self):
        opt = optim.Adam(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = lr_scheduler.ExponentialLR(
            opt, 
            gamma=0.95
            )
        return opt, sched, "Exponential"

    # 5. Adam + CosineAnnealingLR
    def make_cosineannealing(self):
        opt = optim.Adam(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=self.total_epochs
            )
        return opt, sched, "CosineAnnealing"

    # 6. Adam + OneCycleLR
    def make_onecycle(self):
        opt = optim.Adam(
            self.model.parameters(),
            lr=self.lr)
        sched = lr_scheduler.OneCycleLR(
            opt, 
            max_lr=self.lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.total_epochs
            )
        return opt, sched, "OneCycle"

    # 7. Adam + Linear Warmup + Cosine Annealing
    def make_warmupcosine(self):
        opt = optim.Adam(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=5 * self.steps_per_epoch,      # 5エポック分ウォームアップ
            num_training_steps=self.total_epochs * self.steps_per_epoch
        )
        return opt, sched, "WarmupCosine"

    # 8. Schedule-Free Adam
    def make_schedulefree_adam(self):
        opt = AdamWScheduleFreeReference(
            self.model.parameters(), 
            lr=self.lr
            )
        return opt, None, "Schedulefree"
    
    # ==================
    # ===== ADINA ======
    # ==================
    
    # 9 ADINA
    def make_adina(self):
        opt = Adina(
            self.model.parameters(),
            lr=self.lr
            )
        return opt, None, "None"
    
    # 10 Schedule-Free ADINA
    def make_schedulefree_adina(self):
        opt = AdinaSchedulefreeReference(
            self.model.parameters(),
            lr=self.lr
            )
        return opt, None, "Schedulefree"
    
    # 11. Adam + StepLR
    def make_adina_step(self):
        opt = Adina(
            self.model.parameters(), 
            lr=self.lr,
            )
        sched = lr_scheduler.StepLR(
            opt, 
            step_size=5, 
            gamma=0.1
            )
        return opt, sched, "Step"

    # 12. Adam + MultiStepLR
    def make_adina_multistep(self):
        opt = Adina(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = lr_scheduler.MultiStepLR(
            opt, milestones=[5, 10, 15], 
            gamma=0.1
            )
        return opt, sched, "MultiStep"

    # 13. Adam + ExponentialLR
    def make_adina_exponential(self):
        opt = Adina(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = lr_scheduler.ExponentialLR(
            opt, 
            gamma=0.95
            )
        return opt, sched, "Exponential"

    # 14. Adam + CosineAnnealingLR
    def make_adina_cosineannealing(self):
        opt = Adina(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=self.total_epochs
            )
        return opt, sched, "CosineAnnealing"

    # 15. Adam + OneCycleLR
    def make_adina_onecycle(self):
        opt = Adina(
            self.model.parameters(),
            lr=self.lr)
        sched = lr_scheduler.OneCycleLR(
            opt, 
            max_lr=self.lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.total_epochs
            )
        return opt, sched, "OneCycle"

    # 16. Adam + Linear Warmup + Cosine Annealing
    def make_adina_warmupcosine(self):
        opt = Adina(
            self.model.parameters(), 
            lr=self.lr
            )
        sched = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=5 * self.steps_per_epoch,      # 5エポック分ウォームアップ
            num_training_steps=self.total_epochs * self.steps_per_epoch
        )
        return opt, sched, "WarmupCosine"
    
    # 17. Schedule-Free ADINA Approx
    # 近似による計算
    def make_schedulefree_adina_approx(self):
        opt = AdinaSchedulefreeApprox(
            self.model.parameters(),
            lr=self.lr
            )
        return opt, None, "Schedulefree"
    
    # 18. Schedule-Free ADINA LRC
    # 学習率にCを内包
    def make_schedulefree_adina_LRC(self):
        opt = AdinaSchedulefreeLRC(
            self.model.parameters(),
            lr=self.lr
            )
        return opt, None, "Schedulefree"

    # ==================
    # ===== Build ======
    # ==================
    def build_optimizers_and_schedulers(self):

        """
            opt: 最適化手法に指定
            sched: スケジューラの指定
            sched_name: スケジューラのstep時に個体識別で使用

            アルゴリズムの名前をキー、バリューはoptimizer、スケジューラ、スケジューラの名前のタプル
        """

        optimizers = {}

        # 1. Adam (No Scheduler)
        optimizers["Adam"] = self.make_adam()

        # 2. Adam + StepLR
        optimizers['Adam+StepLR'] = self.make_step()

        # 3. Adam + MultiStepLR
        optimizers['Adam+MultiStepLR'] = self.make_multistep()

        # 4. Adam + ExponentialLR
        optimizers['Adam+ExponentialLR'] = self.make_exponential()

        # 5. Adam + CosineAnnealingLR
        optimizers['Adam+CosineAnnealing'] = self.make_cosineannealing()

        # 6. Adam + OneCycleLR
        optimizers['Adam+OneCycleLR'] = self.make_onecycle()

        # 7. Adam + Linear Warmup + Cosine Annealing
        optimizers['Adam+WarmupCosine'] = self.make_warmupcosine()
        
        # 8. Schedule-Free Adam
        optimizers['Schedule-Free Adam'] = self.make_schedulefree_adam()

        """" 
            === ADINA === 
        """

        # 9 ADINA
        optimizers["ADINA"] = self.make_adina()

        # 10 Schedule-Free ADINA
        optimizers["Schedule-Free ADINA"] = self.make_schedulefree_adina()

        # 11. Adam + StepLR
        optimizers['ADINA+StepLR'] = self.make_adina_step()

        # 12. Adam + MultiStepLR
        optimizers['ADINA+MultiStepLR'] = self.make_adina_multistep()

        # 13. Adam + ExponentialLR
        optimizers['ADINA+ExponentialLR'] = self.make_adina_exponential()

        # 14. Adam + CosineAnnealingLR
        optimizers['ADINA+CosineAnnealing'] = self.make_adina_cosineannealing()

        # 15. Adam + OneCycleLR
        optimizers['ADINA+OneCycleLR'] = self.make_adina_onecycle()

        # 16. Adam + Linear Warmup + Cosine Annealing
        optimizers['ADINA+WarmupCosine'] = self.make_adina_warmupcosine()

        # 17. Schedule-Free ADINA Approx
        optimizers["Schedule-Free ADINA Approx"] = self.make_schedulefree_adina_approx()

        # 18. Schedule-Free ADINA LRC
        optimizers["Schedule-Free ADINA LRC"] = self.make_schedulefree_adina_LRC()

        return optimizers
#