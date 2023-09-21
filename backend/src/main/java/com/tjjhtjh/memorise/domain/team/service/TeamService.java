package com.tjjhtjh.memorise.domain.team.service;

import com.tjjhtjh.memorise.domain.team.repository.TeamRepository;
import com.tjjhtjh.memorise.domain.team.service.dto.request.TeamRequest;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class TeamService {

    private final TeamRepository teamRepository;
    private final UserRepository userRepository;
    private static final String NO_USER_EMAIL = "해당 이메일이 존재하지 않습니다";

    @Transactional
    public void createTeam(TeamRequest teamRequest) {
        User user = userRepository.findByEmail(teamRequest.getUserId())
                .orElseThrow(() -> new UsernameNotFoundException(NO_USER_EMAIL));

        if (teamRequest.getName() != null) {
            teamRepository.save(teamRequest.toRegistEntity(user, teamRequest.getName()));
        } else {
            Integer groupNum = teamRepository.findAll().size() + 1;
            String defaultName = user.getNickname() + groupNum;
            teamRepository.save(teamRequest.toRegistEntity(user, defaultName));
        }
    }

}
