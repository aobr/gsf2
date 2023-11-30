subroutine star_star_potential(ms,xs,ys,zs,eps,ns,pots)
use omp_lib
implicit none
! computes the gravitational potential at the set of ns particle positions (xs,ys,zs) 
! from the set of particles themselves

integer*4, intent(in) :: ns ! number of stellar particles
real*8, intent(in), dimension(ns) :: ms,xs,ys,zs ! mass and positions of stellar particles
real*8, intent(in) :: eps ! softening length 
real*8, intent(out), dimension(ns) :: pots ! array holding the potential at all stellar particle positions
integer*4 :: i,j
real*8 :: dist
real*8, dimension(ns) :: ms2,xs2,ys2,zs2

do i=1,ns
  pots(i) = 0.0
  ms2(i) = ms(i)
  xs2(i) = xs(i)
  ys2(i) = ys(i)
  zs2(i) = zs(i)
enddo

!$omp parallel do private(i,j,dist) shared(pots,xs,ys,zs,ms,xs2,ys2,zs2,ms2)
do i=1,ns
  do j=1,ns
    dist = sqrt( (xs(i)-xs2(j))**2 + (ys(i)-ys2(j))**2 + (zs(i)-zs2(j))**2 + eps**2)
    pots(i) = pots(i)+ms2(j)/dist
  enddo
enddo
!$omp end parallel do

do i=1,ns
  pots(i) = pots(i)-ms(i)/eps
enddo

end subroutine star_star_potential


subroutine star_other_potential(m,x,y,z,xs,ys,zs,eps,n,ns,pots)
use omp_lib
implicit none
! computes the gravitational potential at the set of ns particle positions (xs,ys,zs) 
! from another set of n particles (m,x,y,z)

integer*4, intent(in) :: n,ns
real*8, intent(in), dimension(n) :: m,x,y,z
real*8, intent(in), dimension(ns) :: xs,ys,zs
real*8, intent(in) :: eps
real*8, intent(out), dimension(ns) :: pots
integer*4 :: i,j
real*8 :: dist

!$omp parallel do private(i,j,dist) shared(pots,xs,ys,zs,x,y,z,m)
do i=1,ns
  do j=1,n
    dist = sqrt( (xs(i)-x(j))**2 + (ys(i)-y(j))**2 + (zs(i)-z(j))**2 + eps**2)
    pots(i) = pots(i)+m(j)/dist
  enddo
enddo
!$omp end parallel do

end subroutine star_other_potential


subroutine midplane_potential(m,x,y,z,xin,yin,zin,eps,n,ni,pote)
implicit none

integer*4, intent(in) :: n,ni
real*8, intent(in), dimension(n) :: m,x,y,z
real*8, intent(in), dimension(ni) :: xin,yin,zin
real*8, intent(in) :: eps
real*8, intent(out), dimension(ni) :: pote
integer*4 :: i,j
real*8 :: dist
real*8 :: tol=1.0e-6

do i=1,ni
  pote(i) = 0.0
enddo

do i=1,ni
  do j=1,n
    dist = sqrt( (xin(i)-x(j))**2 + (yin(i)-y(j))**2 + (zin(i)-z(j))**2)
    if (dist.gt.tol) then
        pote(i) = pote(i)+m(j)/dist
    endif
  enddo
enddo

end subroutine midplane_potential


subroutine midplane_vcirc2(m,x,y,z,xin,yin,zin,eps,n,ni,vcirc2)
implicit none

integer*4, intent(in) :: n,ni
real*8, intent(in), dimension(n) :: m,x,y,z
real*8, intent(in), dimension(ni) :: xin,yin,zin
real*8, intent(in) :: eps
real*8, intent(out), dimension(ni) :: vcirc2
integer*4 :: i,j
real*8 :: dist32
real*8, dimension(ni) :: acc_x, acc_y, acc_z
real*8 :: tol

tol = 1.0e-6

do i=1,ni
  acc_x(i) = 0.0
  acc_y(i) = 0.0
!   acc_z(i) = 0.0
enddo

do i=1,ni
  do j=1,n
    dist32 = (sqrt( (xin(i)-x(j))**2 + (yin(i)-y(j))**2 + (zin(i)-z(j))**2))**3
    if (dist32.gt.tol) then
        acc_x(i) = acc_x(i)+m(j)/dist32*(xin(i)-x(j))
        acc_y(i) = acc_y(i)+m(j)/dist32*(yin(i)-y(j))
    endif
!     acc_z(i) = acc_z(i)+m(j)/dist32*(zin(i)-z(j))
  enddo
enddo

do i=1,ni
  vcirc2(i) = acc_x(i)*xin(i) + acc_y(i)*yin(i)
enddo    

end subroutine midplane_vcirc2
