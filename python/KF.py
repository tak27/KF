import numpy
import numpy.linalg

class KF:
    def __init__(self,x0,P0,A=None,B=None,G=None,Q=None,H=None,R=None,history=True):
        """
        Constructs a KF object.

        Parameters
        ----------
        x0 : D x 1 numpy.array
            initial state, which must be a vertical vector
        P0 : D x D numpy.array
            initial error covariance matrix for x0
        A : D x D numpy.array
            the state transition matrix
        B : D x E numpy.array
            the control input model
        G : D x D numpy.array
            the process noise transition matrix
        Q : D x D numpy.array
            the covariance of the process noise
        H : C x D numpy.array
            the observation model
        R : C x C numpy.array
            the covariance of the observation noise
        history : bool
        """
        self.x=x0.reshape((x0.size,1))
        self.P=P0
        self.A=A
        self.B=B
        self.G=G
        self.Q=Q
        self.H=H
        self.R=R
        self.history=history
        if self.history:
            self.hist_x=self.x
            self.hist_P=numpy.transpose([numpy.diag(self.P)])
            self.hist_xp=numpy.empty((self.hist_x.shape[0],0))
            self.hist_Pp=numpy.empty((self.hist_P.shape[0],0))

    def predict(self,u=None,A=None,B=None,G=None,Q=None):
        """
        Predicts the state at one step ahead in time

        Parameters
        ----------
        u : E x 1 numpy.array
            control vector
        A : D x D numpy.array
            the state transition matrix
        B : D x E numpy.array
            the control input model
        G : D x D numpy.array
            the process noise transition matrix
        Q : D x D numpy.array
            the covariance of the process noise
        """
        if A is not None:
            self.A=A
        if B is not None:
            self.B=B
        if G is not None:
            self.G=G
        if self.G is None:
            self.G=numpy.eye(A.shape[0],A.shape[1])
        if Q is not None:
            self.Q=Q
        xp=numpy.dot(self.A,self.x.reshape((self.x.size,1)))
        if u is not None and self.B is not None:
            xp=xp+numpy.dot(self.B,u.reshape((u.size,1)))
        Pp=numpy.dot(numpy.dot(self.A,self.P),self.A.T)+numpy.dot(numpy.dot(self.G,self.Q),self.G.T)
        return (xp,Pp)

    def update(self,z,pred=None,H=None,R=None):
        """
        Updates the state with the given observation

        Parameters
        ----------
        z : C x 1 numpy.array
            observation
        H : C x D numpy.array
            the observation model
        R : C x C numpy.array
            the covariance of the observation noise
        """
        z=z.reshape((z.size,1))
        if pred is None:
            pred=self.predict()
        if H is not None:
            self.H=H
        if self.H is None:
            self.H = numpy.eye(z.shape[0],self.A.shape[0])
        if R is not None:
            self.R=R
        xp,Pp=pred[0],pred[1]

        K=numpy.dot(numpy.dot(Pp,self.H.T),numpy.linalg.inv(self.R+numpy.dot(numpy.dot(self.H,Pp),self.H.T)))
        KH=numpy.dot(K,self.H)
        x=xp+numpy.dot(K,(z-numpy.dot(self.H,xp)))
        P=numpy.dot(numpy.eye(KH.shape[0],KH.shape[1])-KH,Pp)

        self.x,self.P=x,P
        if self.history:
            self.hist_x=numpy.append(self.hist_x,self.x,axis=1)
            self.hist_P=numpy.append(self.hist_P,numpy.transpose([numpy.diag(self.P)]),axis=1)
            self.hist_xp=numpy.append(self.hist_xp,xp,axis=1)
            self.hist_Pp=numpy.append(self.hist_Pp,numpy.transpose([numpy.diag(Pp)]),axis=1)
        return (self.x,self.P)

    def eval_dist(self,z,pred=None,H=None):
        if z.shape[1] is None:
            dist=z.reshape((z.size,1))
        if pred is None:
            pred=self.predict()
        if H is None:
            H=self.H
        xp,Pp=pred[0],pred[1]
        dist=z.T
        Hxp=numpy.dot(H,xp)
        for e in dist:
            e[...]=e-Hxp[:,0]
        dist=dist.T
        devi=numpy.dot(H,numpy.sqrt(numpy.transpose(numpy.diag(Pp))))
        devi=devi.reshape((devi.size,1))
        conf=numpy.linalg.norm(dist/devi,ord=2,axis=0)
        conf=conf.reshape((1,conf.size))
        return dist,conf

if __name__ == "__main__":
    import numpy.random
    import matplotlib
    import matplotlib.pyplot as pyplot

    softsensor=True

    T=300
    dt=0.1

    #
    # DYNAMICS
    #

    A=numpy.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    B=numpy.array([
        [dt*dt/2, 0],
        [0, dt*dt/2],
        [dt, 0],
        [0, dt]])
    G=numpy.eye(4)
    Q=0.1*numpy.eye(4)
    H=numpy.eye(4)
    if softsensor:
        R=3*numpy.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2/dt, 0],
            [0, 0, 0, 2/dt]])
    else:
        R=3*numpy.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

    #
    # GROUNDTRUETH GENERATION
    #

    vmean=numpy.zeros((4,1))
    wmean=numpy.zeros((4,1))
    v=vmean+numpy.dot(numpy.linalg.cholesky(Q),numpy.random.randn(A.shape[0],T))
    w=wmean+numpy.dot(numpy.linalg.cholesky(R),numpy.random.randn(H.shape[0],T))

    x=numpy.zeros((A.shape[0],T))
    x[:,0]=numpy.transpose(20*numpy.random.rand(A.shape[1],1))
    u=numpy.random.rand(B.shape[1],T)
    u[0.4<u]=0
    u=u-0.5
    z=numpy.zeros((H.shape[0],T))
    z[:,0]=numpy.dot(H,x[:,0])+w[:,0]
    if softsensor:
        z[2:4,0]=0 # indifferentiable at t=0
    for k in range(1,T):
        x[:,k]=numpy.dot(A,x[:,k-1])+numpy.dot(B,u[:,k-1])+v[:,k-1]
        z[:,k]=numpy.dot(H,x[:,k])+w[:,k]
        if softsensor:
            z[2:4,k]=(z[0:2,k]-z[0:2,k-1])/dt # overwrite velocity with differentiated positions

    #
    # FILTERING
    #

    kf=KF(z[:,0],R,A=A,B=B,G=G,Q=Q,H=H,R=R,history=True)
    for k in range(1,T):
        pred=kf.predict(u[:,k])
        kf.update(z[:,k],pred=pred)

    #
    # PLOT
    #

    # two components of state vector to be plot below
    comp1=0
    comp2=1
    pyplot.plot(x[comp1,:],x[comp2,:],label='$x_k$',color='black',linestyle='solid')
    pyplot.scatter(z[comp1,:],z[comp2,:],label='$z_k$',color='red')
    pyplot.plot(kf.hist_x[comp1,:],kf.hist_x[comp2,:],label='$x_{k|k}$',color='blue')
    pyplot.legend()
    pyplot.show()
