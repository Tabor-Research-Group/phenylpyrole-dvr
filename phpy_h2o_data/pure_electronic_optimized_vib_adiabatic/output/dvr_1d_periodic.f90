!    Program for the calculation of energies and wavefunctions of
!    One dimensional periodic potentials
!
!    The input files are input_period.dat and potential_file.dat
!    The file input_period.dat is automatically built 
!    by executing the Jupyter notebook
!    
!    User needs to supply their own matrix diagonlization subroutine
!    The diagonalization is called house in the current version
!
!    Author Daniel Tabor
!    Dept. of Chemistry, Texas A&M University
!    Last Updated January 17 2022
!    Adapted from the structure of a Cartesian coordinate DVR code written by 
!    Edwin L. Sibert 
!    Dept. of Chemistry and Chemical Biology 
!    University of Wisconsin--Madison
!    Formulation follows the work of Colbert and Miller
!    J. Chem. Phys. 96, 1982 (1992) 
!    https://doi.org/10.1063/1.462100

      program dvr_1d
      implicit none
      real(kind=8), allocatable, dimension(:)    :: angle          !angle_array
      real(kind=8), allocatable, dimension(:)    :: t          !the kinetic energy matrix elements
      real(kind=8), allocatable, dimension(:)    :: bk,root    !root are the eigenvalues; bk is storage
      real(kind=8), allocatable, dimension(:,:)  :: h          !the hamiltonian and then the transformation matrix
      real(kind=8)                               :: con   !used in setting up kinetic energy
      real(kind=8)                               :: xmin,xmax  !min and max value of grid
      real(kind=8)                               :: delx       !grid spacing                
      real(kind=8)                               :: xx,pi      !xx is coordinate value 
      real(kind=8)                               :: v11         !potentials
      real(kind=8)                               :: mass_x     !mass of X in the XH bond
      real(kind=8)                               :: mass_red   !diatomic reduced mass
      real(kind=8)                               :: re         !re used in definition of the spf coordinates.
      real(kind=8)                               :: vmin,xsal  !minimum of the potential
      real(kind=8)                               :: cosine_part, sine_part 
      integer                                    :: i,k,ip       !counters
      integer                                    :: npt        !the total number of basis functions
      integer                                    :: npoints    !the total number of dvr points
      integer                                    :: norder     !the order of the expansion
      integer                                    :: ix,istate,counter
      character(len=40)                               :: filename_out,file_num
      real(kind=8), parameter                    :: clight=2.99792458D10,av=6.0221367D23,hbar=1.05457266D-27 
      real(kind=8), parameter                    :: au_to_cm=219474.63 
!     ^^ conversion factors

!     --------------------------------------------------------------------------------
      pi = dacos(-1.d0)
!     --------------------------------------------------------------------------------
      open (16,file='input_period.dat')
      read(16,*) ! blank line
      read(16,*)npt                    ! Number of states to calculate and print
      read(16,*)xmin                   !the min value of x points
      read(16,*)xmax                   !the max value of x points
      read(16,*)npoints                !the number of dvr points
      read(16,*)mass_x                 !Mass of torsion degree of freedom 
      read(16,*)filename_out                 !Mass of oscillator 

      open (17,file='potential_file.dat')

      allocate(t(0:2*npoints),h(2*npoints+1,2*npoints+1),bk(2*npoints+1),root(2*npoints+1))
      xmin = 0.d0
      xmax = 2.d0*pi
      delx = (xmax-xmin)/dfloat(2*npoints+1)  !grid spacing 

!     Set up angle array

      allocate(angle(1:2*npoints+1))
      do i = 1,2*npoints+1
        angle(i) = xmin + dfloat(i)*delx 
      enddo  
!     --------------------------------------------------------------------------------
!     set up the kinetic energy operator in the dvr !following Colbert and Miller JCP

      mass_red = mass_x  ! 1.d0 
      con = 0.5d0/mass_red !*hbar*av*1.d16/mass_red/(2.d0*pi*clight)
      write(*,*)' the mass prefactor in wavenumbers is ',con
      t(0) = con*dfloat(npoints)*(dfloat(npoints)+1.d0)/3.d0
    
      do i = 1,2*npoints
       cosine_part = dcos((pi*dfloat(i))/(2.d0*dfloat(npoints)+1.d0))
       sine_part = 2.d0*dsin((pi*dfloat(i))/(2.d0*dfloat(npoints)+1.d0))**2
       t(i) = con*((-1.d0)**(dfloat(i)))*cosine_part/sine_part
      enddo
       
      do i = 1,2*npoints+1
       do ip = 1,i
         h(i,ip) = t(iabs(i-ip))
         h(ip,i) = h(i,ip)
       enddo
      enddo
!     add on the potential contribution
      write(*,*) 'adding potential'
!     Read potential
      do i = 1,2*npoints+1
         read(17,*) xx,v11
!    Check for internal consistency
!    from potential file generated by Jupyter notebok
       if (dabs(xx-angle(i)) .lt. 1d-5) then 
        h(i,i) = v11 + h(i,i) 
       else
        write(*,*) 'There is a point mismatch between points'
        write(*,*) 'Fortran program says the value is',angle(i)
        write(*,*) 'Potential file says value is',xx
        stop
       endif  
      enddo 

!     -------------------------------------------------------------------
!     Optional Printing of Hamiltonian for Testing
      write(*,*) 'Here is the start of the ham before diagonalizing' 
      do i = 1,max(8,npt)
        write(*,'(13F10.5)') h(i,:) 
      enddo
!     ------------------------------------------------------------------
!     -------------------------------------------------------------------
!     Below is the matrix diagonlization routine
!     Needs to return eigenvalues (root) and eigenvectors 
!     In the following routine the eigenvectors are returned 
!     In the input matrix H
      call house(h,2*npoints+1,root,bk)

!     Write and print the calculated DVR energies in both atomic units 
!     and wavenumbers. Written to both screen and file for later
!     Jupyter notebook analysis
      open(unit=19,file=trim(filename_out)//'_dvr_energies.out') 
      write(*,*) 'Here are the absolute energy levels'
      do i = 1,min(npt,10)
        write(*,*)i,root(i),root(i)*au_to_cm
      enddo
        
      do i = 1,2*npoints+1
       write(19,*)root(i)
      enddo 

!     DVR wavefunctions (in h) 
!     are written to individual files for later 
!     analysis of overlaps 
      istate = 0
      write(*,*) 'Taking wavefunction slices and writing full wavefunction for state',istate

      do k=1,max(5,npt)
        write(file_num,'(i5)') k
        open(unit=79,file=trim(filename_out)//'_wavefunction_'//trim(adjustl(file_num))//'.dat') 
        counter = 1
        do ix = 1,2*npoints+1
          write(79,*) xmin+delx*ix,h(counter,k)
          counter = counter+1
        enddo
        close(79) 
      enddo

      
!     -------------------------------------------------------------------
      end program dvr_1d
    
!
!    A - The (N,N) matrix to be diagonalized.
!    D - Array of dimension N containing eigenvalues on output.
!    
     subroutine house(a,n,d,e)
     implicit none
     real(kind=8), dimension(n,n)  :: a
     real(kind=8), dimension(n)    :: d,e
     real(kind=8)                  :: b,dd,h,f,g,hh,scale
     real(kind=8)                  :: p,r,s,c
     integer                       :: j,k,i,l,n,iter,m

     if(n>=1)then
      do  i=n,2,-1  
          l=i-1
          h=0.d0
          scale=0.d0
          if(l.gt.1)then
            do k=1,l
              scale=scale+dabs(a(i,k))
            enddo
            if(scale.eq.0.d0)then
              e(i)=a(i,l)
            else
              do k=1,l
                A(I,K)=A(I,K)/SCALE
                H=H+A(I,K)**2
              enddo
              F=A(I,L)
              G=-SIGN(dSQRT(H),F)
              E(I)=SCALE*G
              H=H-F*G
              A(I,L)=F-G
              F=0.d0
              DO J=1,L
                A(J,I)=A(I,J)/H
                G=0.d0
                DO  K=1,J
                  G=G+A(J,K)*A(I,K)
                enddo
                IF(L.GT.J)THEN
                  DO K=J+1,L
                    G=G+A(K,J)*A(I,K)
                  enddo
                ENDIF
                E(J)=G/H
                F=F+E(J)*A(I,J)
              enddo
              HH=F/(H+H)
              DO J=1,L
                F=A(I,J)
                G=E(J)-HH*F
                E(J)=G
                DO K=1,J
                  A(J,K)=A(J,K)-F*E(K)-G*A(I,K)
                enddo
              enddo
            ENDIF
          ELSE
            E(I)=A(I,L)
          ENDIF
          D(I)=H
    enddo
   ENDIF
      D(1)=0.d0
      E(1)=0.d0
      DO 23 I=1,N
        L=I-1
        IF(D(I).NE.0.d0)THEN
          DO 21 J=1,L
            G=0.d0
            DO 19 K=1,L
              G=G+A(I,K)*A(K,J)
19          CONTINUE
            DO 20 K=1,L
              A(K,J)=A(K,J)-G*A(K,I)
20          CONTINUE
21        CONTINUE
        ENDIF
        D(I)=A(I,I)
        A(I,I)=1.d0
        IF(L.GE.1)THEN
          DO 22 J=1,L
            A(I,J)=0.d0
            A(J,I)=0.d0
22        CONTINUE
        ENDIF
23    CONTINUE
!
!  Now diagonalize the triadiagonal matrix produced above.
!
      IF (N.GT.1) THEN
        DO I=2,N
          E(I-1)=E(I)
        enddo
        E(N)=0.d0
        DO 45 L=1,N
          ITER=0
1         DO 42 M=L,N-1
            DD=dABS(D(M))+dABS(D(M+1))
            IF (dABS(E(M))+DD==DD) GO TO 2
42        CONTINUE
          M=N
2         IF(M.NE.L)THEN
            IF(ITER.EQ.30)PAUSE 'too many iterations'
            ITER=ITER+1
            G=(D(L+1)-D(L))/(2.d0*E(L))
            R=SQRT(G**2+1.d0)
            G=D(M)-D(L)+E(L)/(G+SIGN(R,G))
            S=1.d0
            C=1.d0
            P=0.d0
            DO 44 I=M-1,L,-1
              F=S*E(I)
              B=C*E(I)
              IF(dABS(F).GE.dABS(G))THEN
                C=G/F
                R=dSQRT(C**2+1.d0)
                E(I+1)=F*R
                S=1.d0/R
                C=C*S
              ELSE
                S=F/G
                R=dSQRT(S**2+1.d0)
                E(I+1)=G*R
                C=1.d0/R  
                S=S*C
              ENDIF
              G=D(I+1)-P
              R=(D(I)-G)*S+2.d0*C*B
              P=S*R
              D(I+1)=G+P
              G=C*R-B
              DO 43 K=1,N
                F=A(K,I+1)
                A(K,I+1)=S*A(K,I)+C*F
                A(K,I)=C*A(K,I)-S*F
43            CONTINUE
44          CONTINUE
            D(L)=D(L)-P
            E(L)=G
            E(M)=0.d0
            GO TO 1
          ENDIF
45      CONTINUE
      ENDIF
!
!  Now sort the eigenvalues and eigenvectors increasing order.
!
      DO I=1,N-1
        K=I
        P=D(I)
        DO J=I+1,N
          IF(D(J)<=P)THEN
            K=J
            P=D(J)
          ENDIF
        enddo
        IF(K.NE.I)THEN
          D(K)=D(I)
          D(I)=P
          DO J=1,N
            P=A(J,I)
            A(J,I)=A(J,K)
            A(J,K)=P
          enddo
        ENDIF
      enddo

   end subroutine house

	
