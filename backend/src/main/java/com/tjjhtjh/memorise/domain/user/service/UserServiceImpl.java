package com.tjjhtjh.memorise.domain.user.service;

import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.exception.UserEmailDuplicateException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.Role;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;

    @Override
    @Transactional
    public void join(JoinRequest joinRequest) {
         if (userRepository.findByEmailAndIsDeletedFalse(joinRequest.getEmail()).isPresent()) {
             throw new UserEmailDuplicateException("이미 존재하는 이메일입니다.");
         }
         User user = new User(joinRequest.getEmail(), joinRequest.getNickname(), joinRequest.getProfile(), Role.MEMBER);
         userRepository.save(user);
    }

    @Override
    public User getUserInfo(String email) {
        return userRepository.findByEmailAndIsDeletedFalse(email).orElseThrow(() -> new NoUserException("존재하지 않는 유저입니다."));
    }

}
