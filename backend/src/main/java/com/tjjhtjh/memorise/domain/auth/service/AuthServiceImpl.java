package com.tjjhtjh.memorise.domain.auth.service;

import com.tjjhtjh.memorise.domain.auth.service.dto.request.LoginRequest;
import com.tjjhtjh.memorise.domain.auth.service.dto.response.LoginResponse;
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
@Slf4j
public class AuthServiceImpl implements AuthService {

    private final UserRepository userRepository;

    @Override
    public LoginResponse login(LoginRequest loginRequest) {
        User user = userRepository.findByEmail(loginRequest.getEmail())
            .orElseThrow(() -> new NoUserException("존재하지 않는 유저입니다."));
        return new LoginResponse(true, user);
    }
}
