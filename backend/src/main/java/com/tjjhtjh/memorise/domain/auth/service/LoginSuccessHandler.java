package com.tjjhtjh.memorise.domain.auth.service;

import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.global.jwt.service.JwtService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;

@RequiredArgsConstructor
public class LoginSuccessHandler extends SimpleUrlAuthenticationSuccessHandler {

    private final JwtService jwtService;
    private final UserRepository userRepository;

    @Value("${jwt.access.expiration}")
    private String accessTokenExpiration;

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
                                        Authentication authentication) {
        String email = extractUsername(request, authentication);
        String accessToken = jwtService.createAccessToken(email);

        jwtService.sendAccessToken(response, accessToken);
        System.out.println("로그인 성공");
    }

    private String extractUsername(HttpServletRequest request, Authentication authentication) {
        UserDetails userDetails = null;
        try {
            userDetails = (UserDetails) authentication.getPrincipal();
        } catch (Exception e) {
            request.setAttribute("exception", "잘못된 로그인 정보입니다");
        }
        return userDetails.getUsername();
    }
}
