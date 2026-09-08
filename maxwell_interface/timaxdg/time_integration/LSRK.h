#ifndef TIME_INTEGRATION_LSRK_H_
#define TIME_INTEGRATION_LSRK_H_

#include <array>

namespace MaxwellProblem::TimeIntegration {

  /**
   * @brief Low storage Runge-Kutta scheme enum.
   * 
   * Enumerates the 3 different
   * schemes from doi:10.1016/j.jcp.2011.09.003
   * 
   */
  enum LSRK{
    LSRK_12_4,
    LSRK_13_4,
    LSRK_14_4
  };

  /**
   * @brief Data from doi:10.1016/j.jcp.2011.09.003
   * 
   * @tparam S LSRK enum
   */
  template<enum LSRK S>
  struct LSRK_Data;

  /**
   * @brief Table 1 from doi:10.1016/j.jcp.2011.09.003
   * 
   */
  template<>
  struct LSRK_Data<LSRK::LSRK_12_4>
  {
    static constexpr std::array<double, 12> A = {{
      0.0000000000000000,
      -0.0923311242368072,
      -0.9441056581158819,
      -4.3271273247576394,
      -2.1557771329026072,
      -0.9770727190189062,
      -0.7581835342571139,
      -1.7977525470825499,
      -2.6915667972700770,
      -4.6466798960268143,
      -0.1539613783825189,
      -0.5943293901830616
    }};
    
    static constexpr std::array<double, 12> B = {{
      0.0650008435125904,
      0.0161459902249842,
      0.5758627178358159,
      0.1649758848361671,
      0.3934619494248182,
      0.0443509641602719,
      0.2074504268408778,
      0.6914247433015102,
      0.3766646883450449,
      0.0757190350155483,
      0.2027862031054088,
      0.2167029365631842
    }};

    static constexpr std::array<double, 12> c = {{
      0.0000000000000000,
      0.0650008435125904,
      0.0796560563081853,
      0.1620416710085376,
      0.2248877362907778,
      0.2952293985641261,
      0.3318332506149405,
      0.4094724050198658,
      0.6356954475753369,
      0.6806551557645497,
      0.7143773712418350,
      0.9032588871651854
    }};
  };

  /**
   * @brief Table 2 from doi:10.1016/j.jcp.2011.09.003
   * 
   */
  template<>
  struct LSRK_Data<LSRK::LSRK_13_4>
  {
    static constexpr std::array<double, 13> A = {{
      0.0000000000000000,
      -0.6160178650170565,
      -0.4449487060774118,
      -1.0952033345276178,
      -1.2256030785959187,
      -0.2740182222332805,
      -0.0411952089052647,
      -0.1797084899153560,
      -1.1771530652064288,
      -0.4078831463120878,
      -0.8295636426191777,
      -4.7895970584252288,
      -0.6606671432964504
    }};
    
    static constexpr std::array<double, 13> B = {{
      0.0271990297818803,
      0.1772488819905108,
      0.0378528418949694,
      0.6086431830142991,
      0.2154313974316100,
      0.2066152563885843,
      0.0415864076069797,
      0.0219891884310925,
      0.9893081222650993,
      0.0063199019859826,
      0.3749640721105318,
      1.6080235151003195,
      0.0961209123818189
    }};

    static constexpr std::array<double, 13> c = {{
      0.0000000000000000,
      0.0271990297818803,
      0.0952594339119365,
      0.1266450286591127,
      0.1825883045699772,
      0.3737511439063931,
      0.5301279418422206,
      0.5704177433952291,
      0.5885784947099155,
      0.6160769826246714,
      0.6223252334314046,
      0.6897593128753419,
      0.9126827615920843
    }};
  };

  /**
   * @brief Table 3 from doi:10.1016/j.jcp.2011.09.003
   * 
   */
  template<>
  struct LSRK_Data<LSRK::LSRK_14_4>
  {
    static constexpr std::array<double, 14> A = {{
      0.0000000000000000,
      -0.7188012108672410,
      -0.7785331173421570,
      -0.0053282796654044,
      -0.8552979934029281,
      -3.9564138245774565,
      -1.5780575380587385,
      -2.0837094552574054,
      -0.7483334182761610,
      -0.7032861106563359,
      0.0013917096117681,
      -0.0932075369637460,
      -0.9514200470875948,
      -7.1151571693922548
    }};
    
    static constexpr std::array<double, 14> B = {{
      0.0367762454319673,
      0.3136296607553959,
      0.1531848691869027,
      0.0030097086818182,
      0.3326293790646110,
      0.2440251405350864,
      0.3718879239592277,
      0.6204126221582444,
      0.1524043173028741,
      0.0760894927419266,
      0.0077604214040978,
      0.0024647284755382,
      0.0780348340049386,
      5.5059777270269628
    }};

    static constexpr std::array<double, 14> c = {{
      0.0000000000000000,
      0.0367762454319673,
      0.1249685262725025,
      0.2446177702277698,
      0.2476149531070420,
      0.2969311120382472,
      0.3978149645802642,
      0.5270854589440328,
      0.6981269994175695,
      0.8190890835352128,
      0.8527059887098624,
      0.8604711817462826,
      0.8627060376969976,
      0.8734213127600976
    }};
  };

  /**
   * @brief Low storage Runge-Kutte integrator.
   * 
   * This class implements the low storage Runge-Kutta
   * integrator from doi:10.1016/j.jcp.2011.09.003
   * 
   * It solves differential equations of the form
   * y'(t) = F(t,y(t)).
   * 
   * @tparam S LSRK enum
   */
  template<enum LSRK S>
  class LSRK_Integrator
  {
  private:
    const LSRK_Data<S> data{};
  public:

    /**
     * @brief Calculates one integration step.
     * 
     * This routine calculates one integration step.
     * It takes a right-hand side function-object calculating F(t,y(t)).
     * The function-object needs to overload the operator()
     * with the signature (double t, VectorT& vec, VectorT& des).
     * The value t corresponds to the time, vec to y(t) and 
     * des to F(t,y(t)).
     * 
     * The result is stored back in solution.
     * All temporary vectors change values during computation.
     * 
     * @tparam VectorT Vector type
     * @tparam FuncT Function type
     * @param func right-hand side function-object
     * @param solution solution vector
     * @param tmp1 first temporary vector
     * @param tmp2 sencond temporary vector
     * @param time 
     * @param timestep 
     */
    template<typename VectorT, typename FuncT>
    void operator()(FuncT &func, VectorT &solution, VectorT &tmp1, VectorT &tmp2, double time, double timestep){
      tmp1 = 0;

      auto A = data.A.cbegin();
      auto B = data.B.cbegin();
      auto c = data.c.cbegin();
      for(; A != data.A.end(); A++, B++, c++) {
        func(time + (*c)*timestep, solution, tmp2);
        tmp1.sadd(*A, timestep, tmp2);
        solution.sadd(1., *B, tmp1);
      }
    }
  };

}
#endif //TIME_INTEGRATION_LSRK_H_