class Train:
  def __init__(self, model, num_epochs, train_loader, checkpointer, device) -> None:
    self.model = model
    self.num_epochs = num_epochs
    self.train_loader = train_loader
    self.checkpoint = checkpointer
    self.device = device

    self.epoch_losses = []
    self.current_epoch = 1

  def train(self,verbose=True):
    for epoch in range(self.current_epoch, self.num_epochs+1):
      epoch_losses = self.model.get_init_loss_dict()

      num_batches = 0
      for p,m in self.train_loader:
        # Send data to device
        p = p.to(self.device)
        m = m.to(self.device)

        # Perform one training step
        losses = self.model.train_step(p, m)

        # Accumulate losses for logging
        for k in epoch_losses.keys():
            epoch_losses[k] += losses[k]

        num_batches += 1
        if num_batches % 500 == 0:
          print(num_batches)

      # Average losses over epoch
      for k in epoch_losses:
          epoch_losses[k] /= num_batches

      self.epoch_losses.append(epoch_losses)
      self.model.epoch_step()
      if verbose:
        print(f"Epoch: {epoch} ",epoch_losses)

      self.checkpoint.save(epoch, {
        "model_state" : self.model.get_model_state(epoch),
        "epoch_losses" : self.epoch_losses,
        "current_epoch" : epoch
      })

    return self.epoch_losses

  # -1 = load checkpoint of maximum epoch num
  def load_checkpoint(self,epoch_num = -1):
    persist_dict = self.checkpoint.load(epoch_num)
    if persist_dict is None:
      return
    self.epoch_losses = persist_dict["epoch_losses"]
    self.current_epoch = persist_dict["current_epoch"] + 1
    self.model.load_state(persist_dict["model_state"])

